"""Evaluation metrics for time series forecasting and interval estimation."""

import numpy as np


def compute_metrics(predictions: np.ndarray, actuals: np.ndarray) -> dict:
    """Standard regression metrics for point forecasts.

    Parameters
    ----------
    predictions, actuals : ndarray of shape (n,)

    Returns
    -------
    dict with mae, rmse, mape.
    """
    errors = predictions - actuals
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))

    # MAPE: guard against division by zero (common in wind power data)
    nonzero_mask = np.abs(actuals) > 1e-6
    if nonzero_mask.sum() > 0:
        mape = float(np.mean(np.abs(errors[nonzero_mask] / actuals[nonzero_mask])) * 100)
    else:
        mape = float("nan")

    rmse = float(np.sqrt(np.mean(errors ** 2)))

    return {"mae": round(mae, 6), "rmse": round(rmse, 6), "mape": round(mape, 4)}
