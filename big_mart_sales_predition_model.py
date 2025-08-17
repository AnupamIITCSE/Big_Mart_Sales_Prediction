import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

from scipy.stats import randint, uniform
from tqdm import tqdm

import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# ----------------- Paths and constants -----------------
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
TRAIN_PATH = DATA_DIR / "train_v9rqX0R.csv"
TEST_PATH = DATA_DIR / "test_AbJTz2l.csv"
SUBMISSION_PATH = OUTPUT_DIR / "submit_v1_r_100.csv"
BEST_MODEL_PATH = OUTPUT_DIR / "best_model_r_100.pkl"
SEARCH_LOG_PATH = OUTPUT_DIR / "search_log_r_100.csv"

RANDOM_STATE = 42
N_SPLITS = 5
N_ITER_SEARCH = 80  # Increase to 80-120 for more thorough search

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

# ----------------- Preprocessing helpers -----------------
def normalize_fat_content(series: pd.Series) -> pd.Series:
    x = series.astype(str)
    x = x.str.replace("LF", "Low Fat", regex=False)
    x = x.str.replace("low fat", "Low Fat", regex=False)
    x = x.str.replace("reg", "Regular", regex=False)
    return x

def add_outlet_age(df: pd.DataFrame, base_year=2013) -> None:
    df["Outlet_Age"] = base_year - df["Outlet_Establishment_Year"]

def fix_visibility(df: pd.DataFrame, add_flag=True) -> None:
    if add_flag:
        df["Was_Visibility_Zero"] = (df["Item_Visibility"] == 0).astype(int)
    med = df.loc[df["Item_Visibility"] > 0, "Item_Visibility"].median()
    df.loc[df["Item_Visibility"] == 0, "Item_Visibility"] = med

def impute_item_weight_by_item(df: pd.DataFrame) -> None:
    item_median = df.groupby("Item_Identifier")["Item_Weight"].median()
    glob_med = df["Item_Weight"].median()
    mask = df["Item_Weight"].isna()
    df.loc[mask, "Item_Weight"] = df.loc[mask, "Item_Identifier"].map(item_median).fillna(glob_med)

def impute_outlet_size(df: pd.DataFrame) -> None:
    mode_global_ser = df["Outlet_Size"].mode()
    mode_global = mode_global_ser.iloc[0] if not mode_global_ser.empty else None
    by_type = df.groupby("Outlet_Type")["Outlet_Size"].agg(
        lambda s: s.mode().iloc if not s.mode().empty else np.nan
    )
    mask = df["Outlet_Size"].isna()
    df.loc[mask, "Outlet_Size"] = df.loc[mask, "Outlet_Type"].map(by_type)
    df["Outlet_Size"] = df["Outlet_Size"].fillna(mode_global)

def add_logs(df: pd.DataFrame) -> None:
    df["Item_MRP_Log"] = np.log(np.clip(df["Item_MRP"], 1e-6, None))
    df["Item_Visibility_Log1p"] = np.log1p(np.clip(df["Item_Visibility"], 0, None))

def add_item_id_prefix(df: pd.DataFrame) -> None:
    df["Item_Id_Prefix"] = df["Item_Identifier"].astype(str).str[:2]

def build_preprocessor(numeric_features, categorical_features):
    numeric_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, numeric_features),
            ("cat", categorical_tf, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor

rng_global = np.random.RandomState(RANDOM_STATE)

def sample_from_dist(dist):
    return dist.rvs(random_state=rng_global)

def sample_params(space: dict):
    return {k: sample_from_dist(v) for k, v in space.items()}

def manual_random_search(model_name: str, base_model, preprocessor, param_space: dict,
                         X, y, n_splits=5, n_iter=50):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    logs = []
    best_rmse = np.inf
    best_params = None
    best_pipe = None

    trial_bar = tqdm(range(n_iter), desc=f"{model_name} trials", unit="trial")
    for _ in trial_bar:
        params = sample_params(param_space)
        model = base_model.__class__(**{**base_model.get_params(), **params})
        pipe = Pipeline([("pre", preprocessor), ("model", model)])

        fold_rmses = []
        fold_bar = tqdm(enumerate(kf.split(X), start=1),
                        total=n_splits, leave=False,
                        desc=f"{model_name} folds", unit="fold")
        for fold_idx, (tr, va) in fold_bar:
            X_tr, X_va = X.iloc[tr], X.iloc[va]
            y_tr, y_va = y.iloc[tr], y.iloc[va]
            pipe.fit(X_tr, y_tr)
            preds = pipe.predict(X_va)
            score = rmse(y_va, preds)
            fold_rmses.append(score)
            fold_bar.set_postfix({"fold_rmse": f"{score:.4f}"})

        mean_rmse = float(np.mean(fold_rmses))
        std_rmse = float(np.std(fold_rmses))

        logs.append({
            "model": model_name,
            "rmse": mean_rmse,
            "rmse_std": std_rmse,
            **{k.replace("model__", ""): v for k, v in {"model__" + kk: vv for kk, vv in params.items()}.items()}
        })

        trial_bar.set_postfix({"trial_rmse": f"{mean_rmse:.4f}"})

        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_params = params
            best_pipe = pipe

    log_df = pd.DataFrame(logs)
    log_df["rank_test_score"] = log_df["rmse"].rank(method="min")
    return best_pipe, best_params, best_rmse, log_df

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    target = "Item_Outlet_Sales"
    id_cols = ["Item_Identifier", "Outlet_Identifier"]

    required_cols = {
        "Item_Identifier","Item_Weight","Item_Fat_Content","Item_Visibility","Item_Type",
        "Item_MRP","Outlet_Identifier","Outlet_Establishment_Year","Outlet_Size",
        "Outlet_Location_Type","Outlet_Type"
    }
    assert required_cols.issubset(train.columns), "Train schema missing required columns."
    assert required_cols.issubset(test.columns), "Test schema missing required columns."

    # Combine for consistent preprocessing
    train_mark = train.copy()
    test_mark = test.copy()
    train_mark["__is_train__"] = 1
    test_mark["__is_train__"] = 0
    test_mark[target] = np.nan

    full = pd.concat([train_mark, test_mark], axis=0, ignore_index=True)
    full["Item_Fat_Content"] = normalize_fat_content(full["Item_Fat_Content"])
    add_outlet_age(full, base_year=2013)
    fix_visibility(full, add_flag=True)
    impute_item_weight_by_item(full)
    impute_outlet_size(full)
    add_logs(full)

    # ----------- New Feature: Item_Id_Prefix -----------
    add_item_id_prefix(full)

    # Split back
    train_proc = full[full["__is_train__"] == 1].drop(columns="__is_train__")
    test_proc = full[full["__is_train__"] == 0].drop(columns=["__is_train__", target])

    # ----------- Feature lists with new prefix feature -----------
    categorical_features = [
        "Item_Fat_Content",
        "Item_Type",
        "Outlet_Identifier",
        "Outlet_Size",
        "Outlet_Location_Type",
        "Outlet_Type",
        "Item_Id_Prefix",    # new feature added here!
    ]
    numeric_features = [
        "Item_Weight",
        "Item_Visibility",
        "Item_MRP",
        "Outlet_Age",
        "Item_MRP_Log",
        "Item_Visibility_Log1p",
        "Was_Visibility_Zero",
    ]

    # Enforce dtypes
    for col in categorical_features:
        train_proc[col] = train_proc[col].astype(str)
        test_proc[col] = test_proc[col].astype(str)
    for col in numeric_features:
        train_proc[col] = pd.to_numeric(train_proc[col], errors="coerce")
        test_proc[col] = pd.to_numeric(test_proc[col], errors="coerce")

    y = train_proc[target].astype(float)
    X = train_proc.drop(columns=[target])

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # ----------- Define base models (CPU only) -----------
    lgbm_base = lgb.LGBMRegressor(
        objective="regression",
        random_state=RANDOM_STATE
    )
    xgb_base = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        tree_method="hist",
        n_estimators=3000,
        n_jobs=-1
    )
    cat_base = CatBoostRegressor(
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        verbose=False,
        task_type="CPU"
    )

    # ----------- Search spaces -----------
    lgbm_space = {
        "n_estimators": randint(1000, 3000),
        "learning_rate": uniform(0.02, 0.12),
        "num_leaves": randint(31, 127),
        "min_data_in_leaf": randint(20, 200),
        "feature_fraction": uniform(0.6, 0.4),
        "bagging_fraction": uniform(0.6, 0.4),
        "bagging_freq": randint(0, 3),
        "lambda_l1": uniform(0.0, 0.8),
        "lambda_l2": uniform(0.0, 1.2),
    }
    xgb_space = {
        "learning_rate": uniform(0.02, 0.12),
        "max_depth": randint(3, 10),
        "min_child_weight": uniform(1.0, 5.0),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "reg_alpha": uniform(0.0, 0.8),
        "reg_lambda": uniform(0.0, 1.2),
        "gamma": uniform(0.0, 1.0),
    }
    cat_space = {
        "depth": randint(4, 10),
        "learning_rate": uniform(0.006, 0.12),
        "l2_leaf_reg": uniform(1e-3, 10.0),
        "bagging_temperature": uniform(0.0, 1.0),
        "random_strength": uniform(0.0, 2.0),
        "iterations": randint(1000, 3000),
    }

    # ----------- Progress over models -----------
    all_logs = []
    best_overall = (None, np.inf, None)

    models = [
        # ("LightGBM", lgbm_base, lgbm_space),
        # ("XGBoost",  xgb_base,  xgb_space),
        ("CatBoost", cat_base,  cat_space),
    ]

    with tqdm(total=len(models), desc="Models", unit="model") as model_bar:
        for name, base_model, space in models:
            best_pipe, best_params, best_rmse, log_df = manual_random_search(
                model_name=name,
                base_model=base_model,
                preprocessor=preprocessor,
                param_space=space,
                X=X, y=y,
                n_splits=N_SPLITS,
                n_iter=N_ITER_SEARCH
            )
            log_df["timestamp"] = datetime.now().isoformat(timespec="seconds")
            all_logs.append(log_df)
            if best_rmse < best_overall[1]:
                best_overall = (name, best_rmse, best_pipe)
            model_bar.set_postfix({f"{name}_best_rmse": f"{best_rmse:.4f}"})
            model_bar.update(1)

    best_name, best_rmse, best_pipe = best_overall
    print(f"Best model: {best_name} with CV RMSE = {best_rmse:.6f}")

    log_all = pd.concat(all_logs, axis=0, ignore_index=True)
    log_all = log_all.sort_values(["rmse", "rank_test_score"]).reset_index(drop=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_all.to_csv(SEARCH_LOG_PATH, index=False)
    print(f"Saved search log to {SEARCH_LOG_PATH.resolve()}")

    print(f"Fitting best model ({best_name}) on full training data...")
    best_pipe.fit(X, y)

    print("Predicting on test...")
    test_preds = best_pipe.predict(test_proc)
    test_preds = np.clip(test_preds, 0, None)  # clip negatives

    submission = test_proc[id_cols].copy()
    submission[target] = test_preds
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Wrote predictions to {SUBMISSION_PATH.resolve()}")

    joblib.dump(best_pipe, BEST_MODEL_PATH)
    print(f"Saved best model to {BEST_MODEL_PATH.resolve()}")

if __name__ == "__main__":
    main()
