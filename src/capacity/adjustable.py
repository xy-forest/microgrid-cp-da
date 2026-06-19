"""Adjustable capacity estimation from conformal prediction intervals.

The core insight: CP's coverage guarantee propagates to adjustable capacity
bounds.  If the prediction interval [L, U] satisfies P(y in [L, U]) >= 1-α,
then the probability of upward shortfall (y > U) and downward excess (y < L)
are each bounded by α.

Capacity definitions (all values normalised to [0, 1]):
    U_p = max(0, 1 - (y_hat + q_up))    # upward adjustable capacity
    D_p = max(0, (y_hat - q_dn) - 0)    # downward adjustable capacity
    R_p = U_p + D_p                      # total reserve
"""

import numpy as np


def estimate_capacity(
    predictions: np.ndarray,
    q_up: float,
    q_dn: float,
    rating: float = 1.0,
    minimum: float = 0.0,
) -> dict:
    """Compute adjustable capacity from DA-CP intervals.

    Parameters
    ----------
    predictions : ndarray of shape (n,)
        Point forecasts.
    q_up : float
        Upward quantile from DA-CP.
    q_dn : float
        Downward quantile from DA-CP.
    rating : float
        Maximum rated power (normalised, default 1.0).
    minimum : float
        Minimum feasible power (normalised, default 0.0).

    Returns
    -------
    dict with keys:
        cap_up, cap_dn, reserve  (scalar means over all predictions)
        cap_up_series, cap_dn_series  (ndarrays for time-series plotting)
    """
    upper = predictions + q_up
    lower = predictions - q_dn

    cap_up_series = np.clip(rating - upper, 0.0, rating)
    cap_dn_series = np.clip(lower - minimum, 0.0, rating)
    reserve_series = cap_up_series + cap_dn_series

    return {
        "cap_up": float(np.mean(cap_up_series)),
        "cap_dn": float(np.mean(cap_dn_series)),
        "reserve": float(np.mean(reserve_series)),
        "cap_up_series": cap_up_series,
        "cap_dn_series": cap_dn_series,
    }


def violation_analysis(
    predictions: np.ndarray,
    actuals: np.ndarray,
    q_up: float,
    q_dn: float,
    alpha: float,
) -> dict:
    """Check whether CP coverage guarantee holds on the capacity boundary.

    Under CP theory, the total violation rate should be <= alpha, and the
    upward / downward splits should reflect the asymmetric design of DA-CP.

    Parameters
    ----------
    predictions, actuals : ndarray of shape (n,)
    q_up, q_dn : float
        DA-CP quantiles.
    alpha : float
        Target miscoverage rate.

    Returns
    -------
    dict with violation rates and a pass/fail flag.
    """
    upper = predictions + q_up
    lower = predictions - q_dn

    up_violation = actuals > upper
    dn_violation = actuals < lower
    total_violation = up_violation | dn_violation

    return {
        "total_violation_rate": float(np.mean(total_violation)),
        "up_violation_rate": float(np.mean(up_violation)),
        "dn_violation_rate": float(np.mean(dn_violation)),
        "alpha": alpha,
        "within_bound": float(np.mean(total_violation)) <= alpha + 0.02,  # 2% tolerance
    }
