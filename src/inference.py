"""
Inference: load the registered/trained model, build feature vectors from raw
form input, predict a point estimate + MAPIE interval, and generate a LIME
explanation for a single prediction.
"""

import base64
import io
import os

import joblib
import numpy as np
import pandas as pd
import mlflow
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lime
import lime.lime_tabular

MODEL_REGISTRY_NAME = "autoxplain_best_model"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_model():
    """Loads the Production-aliased model from the MLflow registry. Falls back to
    the local joblib copy if the registry is unavailable (e.g. a fresh CI checkout
    with no mlflow.db/mlruns present yet, or the registry being down) - this keeps
    the app runnable without hard-requiring a live MLflow setup at all times.

    Returns (model, source) where source is "mlflow" or "local", so the caller can
    make correct decisions about how to call .predict() (MLflow pyfunc models and
    raw sklearn/xgboost estimators have slightly different calling conventions).
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_REGISTRY_NAME}@production")
        return model, "mlflow"
    except Exception as e:
        print(f"MLflow registry load failed ({e}); falling back to local models/best_model.pkl")
        model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
        return model, "local"


def load_artifacts():
    """Loads everything else needed for inference: scaler, feature order, category
    metadata, LIME background data, and the calibrated MAPIE wrapper.
    """
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    metadata = joblib.load(os.path.join(MODELS_DIR, "category_metadata.pkl"))
    mapie = joblib.load(os.path.join(MODELS_DIR, "mapie.pkl"))
    best_model_name = joblib.load(os.path.join(MODELS_DIR, "best_model_name.pkl"))
    X_train_bg = pd.read_csv(os.path.join(MODELS_DIR, "X_train_background.csv"))[feature_names]
    return {
        "scaler": scaler, "feature_names": feature_names, "metadata": metadata,
        "mapie": mapie, "best_model_name": best_model_name, "X_train_bg": X_train_bg,
    }


def needs_scaling(best_model_name: str) -> bool:
    return best_model_name in ("Ridge", "SVR")


def build_feature_row(form: dict, feature_names: list) -> pd.DataFrame:
    """Constructs a single-row feature vector matching the exact one-hot encoding
    used at training time. `form` values for categorical fields should be the raw
    category name (e.g. "Petrol", "Manual", "Honda") - baseline categories (the
    ones dropped by drop_first) are represented correctly by simply leaving every
    corresponding dummy column at 0.
    """
    row = {name: 0 for name in feature_names}

    row["KM_Driven"] = float(form["km_driven"])
    row["Car_Age"] = float(form["car_age"])
    row["Owner_Number"] = float(form["owner_number"])
    row["Latitude"] = float(form["latitude"])
    row["Longitude"] = float(form["longitude"])

    fuel_col = f"Fuel_{form['fuel']}"
    if fuel_col in row:
        row[fuel_col] = 1

    if form["transmission"] != "Automatic":
        trans_col = f"Trans_{form['transmission']}"
        if trans_col in row:
            row[trans_col] = 1

    brand_col = f"Brand_{form['brand']}"
    if brand_col in row:
        row[brand_col] = 1

    return pd.DataFrame([row], columns=feature_names)


def _predict_log(model, source, X):
    """Handles the calling-convention difference between an MLflow pyfunc model
    and a raw sklearn/xgboost estimator loaded from the local fallback."""
    if source == "mlflow":
        return np.asarray(model.predict(X)).reshape(-1)
    return model.predict(X)


def predict_price(feature_row, model, source, artifacts) -> float:
    X = feature_row.copy()
    if needs_scaling(artifacts["best_model_name"]):
        X = pd.DataFrame(
            artifacts["scaler"].transform(X), columns=artifacts["feature_names"]
        )
    pred_log = _predict_log(model, source, X)[0]
    return float(np.expm1(pred_log))


def predict_interval(feature_row, artifacts):
    """Returns (lower, upper) in rupees using the calibrated MAPIE wrapper.
    MAPIE wraps the (unscaled-or-scaled, matching training) feature space directly,
    so it handles its own internal model call - no separate scaling branch needed
    here since the MAPIE object was calibrated on whichever space the winning
    model actually uses.
    """
    X = feature_row.copy()
    if needs_scaling(artifacts["best_model_name"]):
        X = pd.DataFrame(
            artifacts["scaler"].transform(X), columns=artifacts["feature_names"]
        )
    _, y_interval_log = artifacts["mapie"].predict_interval(X)
    lower = float(np.expm1(y_interval_log[0, 0, 0]))
    upper = float(np.expm1(y_interval_log[0, 1, 0]))
    return lower, upper


def build_lime_explainer(artifacts):
    return lime.lime_tabular.LimeTabularExplainer(
        training_data=artifacts["X_train_bg"].values,
        feature_names=artifacts["feature_names"],
        mode="regression",
        verbose=False,
    )


def explain_prediction(feature_row, model, source, artifacts, explainer) -> str:
    """Returns a base64-encoded PNG of the LIME explanation for this one prediction."""
    feature_names = artifacts["feature_names"]

    def predict_fn(X_raw):
        X = pd.DataFrame(X_raw, columns=feature_names)
        if needs_scaling(artifacts["best_model_name"]):
            X = pd.DataFrame(artifacts["scaler"].transform(X), columns=feature_names)
        pred_log = _predict_log(model, source, X)
        return np.expm1(pred_log)

    explanation = explainer.explain_instance(
        feature_row.values[0], predict_fn, num_features=8
    )
    fig = explanation.as_pyplot_figure()
    fig.set_size_inches(7, 4.5)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
