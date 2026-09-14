"""Shared evaluation helpers - used by both train.py and tests."""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(y_true_actual, y_pred_log) -> dict:
    """Inverse-transforms log-price predictions back to rupees before scoring,
    so reported metrics are in real, readable units."""
    y_pred_actual = np.expm1(y_pred_log)
    y_pred_actual = np.clip(y_pred_actual, a_min=0, a_max=None)

    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true_actual, y_pred_actual))),
        "MAE": float(mean_absolute_error(y_true_actual, y_pred_actual)),
        "R2": float(r2_score(y_true_actual, y_pred_actual)),
    }
