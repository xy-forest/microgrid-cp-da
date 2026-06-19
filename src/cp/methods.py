"""Conformal Prediction methods for distribution-free uncertainty quantification.

Implements four variants:
- Split CP: baseline equal-weight conformal prediction
- DA-CP: Direction-Aware CP (our proposed method)
- Weighted CP: exponentially decayed calibration weights
- Normalized CP: heteroscedasticity-adaptive via conditional std
"""

import numpy as np


def split_cp(
    cal_errors: np.ndarray,
    test_errors: np.ndarray,
    alpha: float = 0.10,
) -> dict:
    """Standard Split Conformal Prediction with symmetric intervals.

    Parameters
    ----------
    cal_errors : ndarray of shape (n_cal,)
        Absolute prediction errors on the calibration set.
    test_errors : ndarray of shape (n_test,)
        Absolute prediction errors on the test set.
    alpha : float
        Target miscoverage rate (1 - alpha = desired coverage).

    Returns
    -------
    dict with keys: q, coverage, interval_width.
    """
    q = float(np.quantile(cal_errors, 1 - alpha))
    covered = test_errors <= q
    return {
        "q": q,
        "coverage": float(np.mean(covered)),
        "interval_width": float(2 * q),
    }


def da_cp(
    cal_predictions: np.ndarray,
    cal_actuals: np.ndarray,
    test_predictions: np.ndarray,
    test_actuals: np.ndarray,
    alpha: float = 0.10,
) -> dict:
    """Direction-Aware Conformal Prediction (DA-CP).

    Asymmetric coverage: the upward side (risk of shortfall) uses a stricter
    alpha/2, while the downward side (risk of curtailment) uses alpha.

    Parameters
    ----------
    cal_predictions, cal_actuals : ndarray of shape (n_cal,)
    test_predictions, test_actuals : ndarray of shape (n_test,)
    alpha : float
        Target miscoverage rate.

    Returns
    -------
    dict with keys:
        q_up, q_dn, coverage, up_violation, dn_violation,
        cap_up, cap_dn, reserve
    """
    cal_err = np.abs(cal_predictions - cal_actuals)

    # Partition calibration errors by direction
    up_mask = cal_predictions < cal_actuals
    dn_mask = cal_predictions > cal_actuals

    up_errors = cal_err[up_mask] if up_mask.sum() > 0 else cal_err
    dn_errors = cal_err[dn_mask] if dn_mask.sum() > 0 else cal_err

    q_up = float(np.quantile(up_errors, 1 - alpha / 2))
    q_dn = float(np.quantile(dn_errors, 1 - alpha))

    # Coverage and violation rates on test set
    upper_bound = test_predictions + q_up
    lower_bound = test_predictions - q_dn
    covered = (test_actuals <= upper_bound) & (test_actuals >= lower_bound)
    up_violation = test_actuals > upper_bound
    dn_violation = test_actuals < lower_bound

    # Adjustable capacity (clipped to [0, 1] for normalised power)
    cap_up = float(np.clip(1.0 - upper_bound, 0.0, 1.0).mean())
    cap_dn = float(np.clip(lower_bound - 0.0, 0.0, 1.0).mean())

    return {
        "q_up": q_up,
        "q_dn": q_dn,
        "coverage": float(np.mean(covered)),
        "up_violation": float(np.mean(up_violation)),
        "dn_violation": float(np.mean(dn_violation)),
        "cap_up": cap_up,
        "cap_dn": cap_dn,
        "reserve": float(q_up + q_dn),
    }


def weighted_cp(
    cal_errors: np.ndarray,
    decay: float = 0.05,
    alpha: float = 0.10,
) -> dict:
    """Split CP with exponentially decayed calibration weights.

    More recent calibration points (later indices) receive higher weight.
    We observed decay > 0.01 degrades coverage; this is reported as a
    negative result in the paper.

    Parameters
    ----------
    cal_errors : ndarray of shape (n_cal,)
    decay : float
        Exponential decay rate per step.
    alpha : float
        Target miscoverage rate.

    Returns
    -------
    dict with q_weighted and q_equal for comparison.
    """
    n = len(cal_errors)
    weights = np.exp(-decay * np.arange(n - 1, -1, -1))

    # Weighted quantile via sorting
    order = np.argsort(cal_errors)
    sorted_err = cal_errors[order]
    sorted_w = weights[order]
    cum_w = np.cumsum(sorted_w)
    cutoff = cum_w[-1] * (1 - alpha)
    idx = np.searchsorted(cum_w, cutoff)
    q_weighted = float(sorted_err[min(idx, n - 1)])

    q_equal = float(np.quantile(cal_errors, 1 - alpha))

    return {"q_weighted": q_weighted, "q_equal": q_equal}


def normalized_cp(
    cal_predictions: np.ndarray,
    cal_actuals: np.ndarray,
    test_predictions: np.ndarray,
    test_actuals: np.ndarray,
    n_bins: int = 10,
    alpha: float = 0.10,
) -> dict:
    """Normalized CP using bin-based conditional standard deviation.

    Divides the prediction space into n_bins, estimates sigma(x) per bin,
    and normalises residuals.  We observed modest improvement in conditional
    coverage uniformity (+0-0.2 pp).

    Parameters
    ----------
    cal_predictions, cal_actuals : ndarray of shape (n_cal,)
    test_predictions, test_actuals : ndarray of shape (n_test,)
    n_bins : int
        Number of equal-width bins over the prediction range.
    alpha : float
        Target miscoverage rate.

    Returns
    -------
    dict with keys: q_abs, q_norm, cov_abs, cov_norm, conditional_coverage
    """
    cal_errs = np.abs(cal_predictions - cal_actuals)
    test_errs = np.abs(test_predictions - test_actuals)
    default_sigma = float(np.std(cal_errs))

    # Bin calibration errors by prediction value
    bin_edges = np.linspace(
        cal_predictions.min(), cal_predictions.max(), n_bins + 1
    )
    bin_sigmas = {}
    for i in range(n_bins):
        mask = (cal_predictions >= bin_edges[i]) & (cal_predictions < bin_edges[i + 1])
        if mask.sum() >= 5:
            bin_sigmas[i] = float(np.std(cal_errs[mask]))
        else:
            bin_sigmas[i] = default_sigma

    # Assign sigma to every calibration / test point
    def _assign_sigma(preds, bin_edges, bin_sigmas, default_sigma):
        sigma = np.full_like(preds, default_sigma)
        for i in range(len(bin_edges) - 1):
            mask = (preds >= bin_edges[i]) & (preds < bin_edges[i + 1])
            sigma[mask] = bin_sigmas[i]
        return sigma

    cal_sigma = _assign_sigma(cal_predictions, bin_edges, bin_sigmas, default_sigma)
    test_sigma = _assign_sigma(test_predictions, bin_edges, bin_sigmas, default_sigma)

    # Absolute (equal-width) CP
    q_abs = float(np.quantile(cal_errs, 1 - alpha))
    cov_abs = float(np.mean(test_errs <= q_abs))

    # Normalised CP
    norm_scores = cal_errs / (cal_sigma + 1e-8)
    q_norm = float(np.quantile(norm_scores, 1 - alpha))
    margins = q_norm * test_sigma
    cov_norm = float(np.mean(test_errs <= margins))

    # Conditional coverage by ground-truth power level
    power_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 2.0]
    cond_cov = {}
    for j in range(len(power_bins) - 1):
        mask = (test_actuals >= power_bins[j]) & (test_actuals < power_bins[j + 1])
        if mask.sum() < 5:
            continue
        label = f"{power_bins[j]:.1f}-{power_bins[j+1]:.1f}"
        cond_cov[label] = {
            "n": int(mask.sum()),
            "cov_abs": float(np.mean(test_errs[mask] <= q_abs)),
            "cov_norm": float(np.mean(test_errs[mask] <= margins[mask])),
        }

    return {
        "q_abs": q_abs,
        "q_norm": round(float(q_norm), 4),
        "cov_abs": cov_abs,
        "cov_norm": cov_norm,
        "conditional_coverage": cond_cov,
    }
