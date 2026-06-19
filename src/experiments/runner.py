"""Experiment runner: model training, CP evaluation, and capacity verification.

Orchestrates the full pipeline:
    1. Build sliding windows from cleaned data
    2. Train forecast models (LSTM / Transformer / PatchTST)
    3. Evaluate Split CP and DA-CP on each model
    4. Run ablation studies (window length, calibration fraction)
    5. Verify adjustable capacity violation rates
"""

import json
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from ..models.forecasters import LSTMForecaster, TransformerForecaster, PatchTSTForecaster
from ..cp.methods import da_cp
from ..capacity.adjustable import estimate_capacity, violation_analysis
from ..utils.metrics import compute_metrics


# ── Global configuration ──────────────────────────────────────────────

SEQ_LEN = 60
HORIZON = 10
BATCH_SIZE = 128
MAX_EPOCHS = 30
LR = 1e-3
PATIENCE = 5
N_SAMPLES = 15000
ALPHAS = [0.05, 0.10, 0.15, 0.20]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")


# ── Data pipeline ─────────────────────────────────────────────────────

def build_windows(
    csv_path: Path,
    target_col: str,
    n_samples: int = N_SAMPLES,
    seq_len: int = SEQ_LEN,
    horizon: int = HORIZON,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct sliding-window tensors from a time-series CSV.

    Each window uses three channels: the target series, hour-of-day, and
    weekday, each normalised to [0, 1].

    Parameters
    ----------
    csv_path : Path
    target_col : str
        Column name of the target variable.
    n_samples : int
        Number of windows to sample (without replacement).
    seq_len : int
        Lookback window length in minutes.
    horizon : int
        Forecast horizon in minutes.

    Returns
    -------
    X : ndarray of shape (n_samples, seq_len, 3)
    y : ndarray of shape (n_samples, 1)
    """
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    series = df[target_col].values.astype(np.float32)
    idx = pd.DatetimeIndex(df.index)
    hours = (idx.hour / 23.0).values.astype(np.float32)
    weekdays = (idx.weekday / 6.0).values.astype(np.float32)

    total = len(series) - seq_len - horizon
    rng = np.random.RandomState(seed)
    ids = sorted(rng.choice(total, min(n_samples, total), replace=False))

    X_list, y_list = [], []
    for i in ids:
        window = np.stack(
            [series[i : i + seq_len], hours[i : i + seq_len], weekdays[i : i + seq_len]],
            axis=-1,
        )
        target = series[i + seq_len + horizon - 1]
        X_list.append(window)
        y_list.append(target)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32).reshape(-1, 1)


# ── Data splitting & standardisation ──────────────────────────────────

def _split_train_cal_test(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """70% train, 10% calibration, 20% test split."""
    n = len(X)
    n_tr = int(n * 0.7)
    n_cal = int(n * 0.1)
    n_cut = n_tr + n_cal
    return X[:n_tr], y[:n_tr], X[n_tr:n_cut], y[n_tr:n_cut], X[n_cut:], y[n_cut:]


def _standardise_splits(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
) -> tuple:
    """Fit StandardScaler on training data, transform all splits.

    Returns tensors suitable for DataLoader consumption.
    """
    sc_X = StandardScaler().fit(X_tr.reshape(-1, X_tr.shape[-1]))
    sc_y = StandardScaler().fit(y_tr)

    def _prepare(xi, yi):
        B, T, F = xi.shape
        xi_t = torch.FloatTensor(sc_X.transform(xi.reshape(-1, F)).reshape(B, T, F))
        yi_t = torch.FloatTensor(sc_y.transform(yi).flatten())
        return xi_t, yi_t

    Xt, yt = _prepare(X_tr, y_tr)
    Xc, yc = _prepare(X_cal, y_cal)
    Xe, ye = _prepare(X_te, y_te)
    return Xt, yt, Xc, yc, Xe, ye, sc_y


# ── Training loop ─────────────────────────────────────────────────────

def _train_one_epoch(model, loader, optim, loss_fn) -> float:
    """Run one training epoch; return average loss."""
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optim.zero_grad()
        loss = loss_fn(model(xb).squeeze(), yb)
        loss.backward()
        optim.step()
        total_loss += loss.item() * len(xb)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def _validate_one_epoch(model, loader, loss_fn) -> float:
    """Run one validation epoch; return average loss."""
    model.eval()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        total_loss += loss_fn(model(xb).squeeze(), yb).item() * len(xb)
    return total_loss / len(loader.dataset)


def _train_loop(model, tr_loader, cal_loader, max_epochs, patience) -> float:
    """Train with early stopping; return elapsed seconds."""
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.L1Loss()

    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0
    t_start = time.time()

    for _epoch in range(max_epochs):
        _train_one_epoch(model, tr_loader, optim, loss_fn)
        val_loss = _validate_one_epoch(model, cal_loader, loss_fn)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    model.load_state_dict(best_weights)
    return time.time() - t_start


@torch.no_grad()
def _predict(model, X_te, y_te, sc_y) -> tuple[np.ndarray, np.ndarray]:
    """Generate predictions on the test set, inverse-transform to original scale."""
    model.eval()
    preds, actuals = [], []
    for xb, yb in DataLoader(TensorDataset(X_te, y_te), BATCH_SIZE):
        xb = xb.to(DEVICE)
        p = model(xb).squeeze().cpu().numpy()
        preds.extend(p)
        actuals.extend(yb.cpu().numpy())

    preds_raw = sc_y.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    actuals_raw = sc_y.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()
    return preds_raw, actuals_raw


# ── Main training entry point ─────────────────────────────────────────

def train_model(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
    """Train a forecasting model and return predictions plus metrics.

    Parameters
    ----------
    model : nn.Module
    X : ndarray of shape (N, T, F)
    y : ndarray of shape (N, 1)

    Returns
    -------
    dict with keys: metrics, predictions, actuals, elapsed_s, params.
    """
    X_tr, y_tr, X_cal, y_cal, X_te, y_te = _split_train_cal_test(X, y)
    Xt, yt, Xc, yc, Xe, ye, sc_y = _standardise_splits(
        X_tr, y_tr, X_cal, y_cal, X_te, y_te
    )

    tr_loader = DataLoader(TensorDataset(Xt, yt), BATCH_SIZE, shuffle=True)
    cal_loader = DataLoader(TensorDataset(Xc, yc), BATCH_SIZE)

    model = model.to(DEVICE)
    elapsed = _train_loop(model, tr_loader, cal_loader, MAX_EPOCHS, PATIENCE)

    preds, actuals = _predict(model, Xe, ye, sc_y)
    n_params = sum(p.numel() for p in model.parameters())

    return {
        "metrics": compute_metrics(preds, actuals),
        "predictions": preds,
        "actuals": actuals,
        "elapsed_s": round(elapsed, 1),
        "params": n_params,
    }


# ── Experiment orchestrator ───────────────────────────────────────────

def run_all_experiments(data_dir: Path, results_dir: Path) -> dict:
    """Execute the complete experiment suite and save results to JSON.

    A-group: model comparison (LSTM / Transformer / PatchTST)
    B-group: DA-CP evaluation over 4 alpha values
    C-group: window-length ablation
    D-group: capacity violation verification
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict = {}

    for ds_name, csv_file, target in [
        ("wind", "wind_clean.csv", "wind_power"),
        ("demand", "demand_clean.csv", "demand"),
    ]:
        print(f"\n{'='*60}\n  {ds_name}\n{'='*60}")
        all_results[ds_name] = {"models": {}, "cp": {}, "ablation": {}, "capacity": {}}

        X, y = build_windows(data_dir / csv_file, target)

        # ── A-group: model comparison ──
        for model_name, ModelClass in [
            ("LSTM", LSTMForecaster),
            ("Transformer", TransformerForecaster),
            ("PatchTST", PatchTSTForecaster),
        ]:
            print(f"  Training {model_name} ...")
            result = train_model(ModelClass(), X, y)
            all_results[ds_name]["models"][model_name] = {
                k: v
                for k, v in result.items()
                if k not in ("predictions", "actuals")
            }
            print(
                f"    MAE={result['metrics']['mae']:.6f}  "
                f"RMSE={result['metrics']['rmse']:.6f}  "
                f"{result['elapsed_s']:.0f}s  {result['params']:,} params"
            )

            # ── B-group: DA-CP evaluation ──
            preds = result["predictions"]
            acts = result["actuals"]
            n_cal = len(preds) // 5
            cal_p, cal_a = preds[:n_cal], acts[:n_cal]
            tst_p, tst_a = preds[n_cal:], acts[n_cal:]

            cp_results = {}
            for alpha in ALPHAS:
                da = da_cp(cal_p, cal_a, tst_p, tst_a, alpha=alpha)
                cp_results[f"alpha_{alpha}"] = {
                    "coverage": round(da["coverage"] * 100, 2),
                    "q_up": round(da["q_up"], 6),
                    "q_dn": round(da["q_dn"], 6),
                    "up_violation": round(da["up_violation"] * 100, 2),
                    "dn_violation": round(da["dn_violation"] * 100, 2),
                    "cap_up": round(da["cap_up"], 4),
                    "cap_dn": round(da["cap_dn"], 4),
                }
            all_results[ds_name]["cp"][model_name] = cp_results

            # ── D-group: capacity verification (alpha=0.10) ──
            if model_name == "LSTM":
                da = da_cp(cal_p, cal_a, tst_p, tst_a, alpha=0.10)
                viol = violation_analysis(tst_p, tst_a, da["q_up"], da["q_dn"], 0.10)
                cap = estimate_capacity(tst_p, da["q_up"], da["q_dn"])
                all_results[ds_name]["capacity"][model_name] = {
                    "violation_rate": round(viol["total_violation_rate"] * 100, 2),
                    "up_violation": round(viol["up_violation_rate"] * 100, 2),
                    "dn_violation": round(viol["dn_violation_rate"] * 100, 2),
                    "cap_up": round(cap["cap_up"], 4),
                    "cap_dn": round(cap["cap_dn"], 4),
                    "reserve": round(cap["reserve"], 4),
                }

        # ── C-group: window-length ablation ──
        print("  Window-length ablation ...")
        all_results[ds_name]["ablation"]["windows"] = {}
        for seq in [30, 60, 90, 120]:
            Xa, ya = build_windows(
                data_dir / csv_file, target, n_samples=N_SAMPLES // 2, seq_len=seq
            )
            result = train_model(LSTMForecaster(), Xa, ya)
            all_results[ds_name]["ablation"]["windows"][str(seq)] = round(
                result["metrics"]["mae"], 6
            )
            print(f"    window={seq}: MAE={result['metrics']['mae']:.6f}")

    # Save
    out_path = results_dir / "all_experiments.json"
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {out_path}")
    return all_results
