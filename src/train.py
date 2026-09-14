"""
Model training: ML-ready dataset -> tuned models, tracked in MLflow, best model
registered, MAPIE-calibrated for prediction intervals.

Run directly: python -m src.train
"""

import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from mapie.regression import SplitConformalRegressor

from src.evaluate import compute_metrics

RANDOM_STATE = 42
MODEL_REGISTRY_NAME = "autoxplain_best_model"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"  # database-backed store, required for the Model Registry
MODELS_DIR = "models"


def get_model_configs():
    """Model + tuning-grid definitions, kept in one place so train.py, and any
    future retraining script, share exactly the same search spaces."""
    return {
        "Ridge": {
            "estimator": Ridge(random_state=RANDOM_STATE),
            "grid": {"alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]},
            "search_type": "grid",
            "needs_scaling": True,
        },
        "Random Forest": {
            "estimator": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            "grid": {
                "n_estimators": [200, 300, 500],
                "max_depth": [None, 8, 12, 16],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", 0.6, 1.0],
            },
            "search_type": "random",
            "n_iter": 20,
            "needs_scaling": False,
        },
        "XGBoost": {
            "estimator": XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            "grid": {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [3, 4, 5, 6, 8],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
            },
            "search_type": "random",
            "n_iter": 25,
            "needs_scaling": False,
        },
        "SVR": {
            "estimator": SVR(),
            "grid": {
                "C": [0.1, 1, 10, 50, 100],
                "epsilon": [0.01, 0.05, 0.1, 0.2],
                "gamma": ["scale", "auto", 0.01, 0.1],
                "kernel": ["rbf"],
            },
            "search_type": "random",
            "n_iter": 25,
            "needs_scaling": True,
        },
    }


def tune_model(name, config, X_train, X_train_scaled, y_train_log, cv=5):
    X = X_train_scaled if config["needs_scaling"] else X_train
    if config["search_type"] == "grid":
        search = GridSearchCV(config["estimator"], config["grid"], cv=cv, scoring="r2", n_jobs=-1)
    else:
        search = RandomizedSearchCV(
            config["estimator"], config["grid"], n_iter=config["n_iter"], cv=cv,
            scoring="r2", random_state=RANDOM_STATE, n_jobs=-1,
        )
    search.fit(X, y_train_log)
    return search.best_estimator_, search.best_params_


def train_all_models(df: pd.DataFrame, cv=5):
    """Trains, tunes, and MLflow-tracks every model in get_model_configs().
    Returns (results_df, trained_models, best_params, split_data) for downstream use
    (registration, MAPIE calibration, etc.)."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("autoxplain")

    X = df.drop(columns=["Price_INR"])
    y = df["Price_INR"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    y_train_log = np.log1p(y_train)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names, index=X_test.index)

    configs = get_model_configs()
    results = {}
    trained_models = {}
    best_params = {}
    run_ids = {}

    for name, config in configs.items():
        with mlflow.start_run(run_name=name) as run:
            model, params = tune_model(name, config, X_train, X_train_scaled, y_train_log, cv=cv)
            best_params[name] = params
            trained_models[name] = model

            X_eval = X_test_scaled if config["needs_scaling"] else X_test
            y_pred_log = model.predict(X_eval)
            metrics = compute_metrics(y_test, y_pred_log)
            results[name] = metrics

            mlflow.log_params(params)
            mlflow.log_param("needs_scaling", config["needs_scaling"])
            mlflow.log_metrics(metrics)

            # XGBoost's native Booster type isn't in mlflow's sklearn-flavor skops
            # allowlist - use the dedicated xgboost flavor for it, sklearn flavor
            # for everything else.
            if name == "XGBoost":
                mlflow.xgboost.log_model(model, name="model")
            else:
                mlflow.sklearn.log_model(model, name="model")
            run_ids[name] = run.info.run_id

            print(f"{name}: RMSE={metrics['RMSE']:,.0f}  MAE={metrics['MAE']:,.0f}  R2={metrics['R2']:.4f}")

    results_df = pd.DataFrame(results).T.sort_values("R2", ascending=False)

    split_data = {
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled,
        "y_train_log": y_train_log, "feature_names": feature_names, "scaler": scaler,
    }
    return results_df, trained_models, best_params, run_ids, split_data


def register_best_model(results_df, run_ids):
    """Registers the best run's model into the MLflow Model Registry and points
    the 'production' alias at it.

    Uses aliases rather than the older stage-based API (Staging/Production/Archived),
    which MLflow has deprecated in favor of aliases as of 2.9 - this keeps the
    registry usage current rather than shipping an already-legacy pattern.
    """
    best_name = results_df["R2"].idxmax()
    best_run_id = run_ids[best_name]
    model_uri = f"runs:/{best_run_id}/model"

    result = mlflow.register_model(model_uri, MODEL_REGISTRY_NAME)

    client = MlflowClient()
    client.set_registered_model_alias(
        name=MODEL_REGISTRY_NAME, alias="production", version=result.version
    )
    print(f"Registered '{best_name}' as {MODEL_REGISTRY_NAME} v{result.version}, alias=production")
    return best_name, result.version


def calibrate_with_mapie(best_name, configs, split_data, best_params, calib_frac=0.2, alpha=0.1):
    """Refits the winning model's architecture on (train - calibration), calibrates
    a MapieRegressor on the held-out calibration split, and checks empirical
    coverage on the untouched test set.

    A separate calibration split (not the test set) keeps the final test-set metrics
    honest - MAPIE's calibration set must be disjoint from whatever you use to
    validate coverage, or the coverage estimate is optimistically biased.
    """
    config = configs[best_name]
    X_train, y_train_log = split_data["X_train"], split_data["y_train_log"]
    X_train_scaled = split_data["X_train_scaled"]
    X_test, y_test = split_data["X_test"], split_data["y_test"]
    X_test_scaled = split_data["X_test_scaled"]

    X_fit = X_train_scaled if config["needs_scaling"] else X_train
    X_fit_sub, X_calib, y_fit_sub, y_calib = train_test_split(
        X_fit, y_train_log, test_size=calib_frac, random_state=RANDOM_STATE
    )

    final_model = config["estimator"].__class__(**best_params[best_name])
    final_model.fit(X_fit_sub, y_fit_sub)

    mapie = SplitConformalRegressor(
        estimator=final_model, confidence_level=1 - alpha, prefit=True
    )
    mapie.conformalize(X_calib, y_calib)

    X_test_eval = X_test_scaled if config["needs_scaling"] else X_test
    y_pred_log, y_interval_log = mapie.predict_interval(X_test_eval)

    lower = np.expm1(y_interval_log[:, 0, 0])
    upper = np.expm1(y_interval_log[:, 1, 0])
    covered = (y_test.values >= lower) & (y_test.values <= upper)
    coverage = covered.mean()

    print(f"MAPIE calibrated on {best_name} - target coverage {1 - alpha:.0%}, "
          f"empirical coverage on test set: {coverage:.1%}")

    return mapie, coverage


def save_local_artifacts(best_name, trained_models, mapie, split_data, metadata):
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(trained_models[best_name], os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(split_data["scaler"], os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(split_data["feature_names"], os.path.join(MODELS_DIR, "feature_names.pkl"))
    joblib.dump(mapie, os.path.join(MODELS_DIR, "mapie.pkl"))
    joblib.dump(metadata, os.path.join(MODELS_DIR, "category_metadata.pkl"))
    joblib.dump(best_name, os.path.join(MODELS_DIR, "best_model_name.pkl"))

    split_data["X_train"][split_data["feature_names"]].to_csv(
        os.path.join(MODELS_DIR, "X_train_background.csv"), index=False
    )
    print(f"Saved local fallback artifacts to {MODELS_DIR}/")


def run_training_pipeline(ml_ready_csv_path: str, metadata: dict, cv=5):
    """End-to-end: load data -> train/tune/track -> register best -> calibrate MAPIE
    -> save local fallback artifacts. `metadata` is the dict returned by
    features.encode_categorical (category lists), saved alongside the models so
    inference-time code never has to guess at encoding choices.
    """
    df = pd.read_csv(ml_ready_csv_path)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    results_df, trained_models, best_params, run_ids, split_data = train_all_models(df, cv=cv)
    print("\n", results_df, "\n")

    best_name, version = register_best_model(results_df, run_ids)

    configs = get_model_configs()
    mapie, coverage = calibrate_with_mapie(best_name, configs, split_data, best_params)

    save_local_artifacts(best_name, trained_models, mapie, split_data, metadata)

    return {
        "results_df": results_df, "best_name": best_name, "version": version,
        "coverage": coverage,
    }


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/cardekho_kolkata_ml_ready.csv"
    meta_path = sys.argv[2] if len(sys.argv) > 2 else "models/category_metadata.pkl"
    metadata = joblib.load(meta_path) if os.path.exists(meta_path) else {}
    run_training_pipeline(csv_path, metadata)
