"""Generate publication-quality figures for the course paper.

Uses REAL experiment data — no simulated data.
Data sources:
  - Prediction CSVs: 实验包结果/results/{wind,demand}_{lstm,transformer,patchtst}_*.csv
  - all_experiments.json: model metrics, CP results, capacity verification
  - patch_results.json: ARIMA baseline, LSTM repeats, CP head-to-head, window ablation
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.family": "serif",
})

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Data sources — REAL experiment outputs
RES_DIR = Path("/Users/xinyan/Desktop/科研/算法分析设计大作业/实验包结果/results")
PATCH = Path("/Users/xinyan/Desktop/科研/算法分析设计大作业/实验补丁包/results/patch_results.json")
DATA_DIR = Path("/Users/xinyan/Desktop/科研/算法分析设计大作业/microgrid_forecast/data")

with open(RES_DIR / "all_experiments.json") as f:
    main = json.load(f)
with open(PATCH) as f:
    patch = json.load(f)


def save(name):
    plt.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    print(f"  -> {FIG_DIR / name}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Fig 1: Method framework overview (schematic — kept as-is, it's a diagram)
# ═══════════════════════════════════════════════════════════════════════
def fig1_framework():
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    boxes = [
        (0.05, 0.55, "Historical Data\n(wind, demand)\n60-min windows", "#E8EAF6"),
        (0.38, 0.55, "Forecasting Models\nLSTM / Transformer / PatchTST\n-> point prediction y_hat", "#C8E6C9"),
        (0.71, 0.55, "Direction-Aware CP (DA-CP)\na_up=a/2, a_down=a\n-> [y_hat-q_down, y_hat+q_up]", "#FFE0B2"),
        (0.38, 0.10, "Adjustable Capacity\nU=clip(1-U_bound,0,1)\nD=clip(L_bound,0,1)\nViolation <= alpha", "#FFCDD2"),
    ]
    for x, y, text, color in boxes:
        ax.text(x + 0.12, y + 0.15, text, ha="center", va="center",
                fontsize=8, bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.9),
                transform=ax.transAxes)
    for p in [(0.27, 0.70, 0.38, 0.70), (0.56, 0.70, 0.71, 0.70), (0.50, 0.55, 0.50, 0.42)]:
        ax.annotate("", xy=(p[2], p[3]), xytext=(p[0], p[1]),
                    arrowprops=dict(arrowstyle="->", lw=1.5), transform=ax.transAxes)
    ax.set_title("Fig.1  Method framework: forecasting -> DA-CP -> adjustable capacity", y=0.98)
    save("fig1_framework.png")


# ═══════════════════════════════════════════════════════════════════════
# Fig 2: One-week time series from REAL cleaned data
# ═══════════════════════════════════════════════════════════════════════
def fig2_timeseries():
    wind = pd.read_csv(DATA_DIR / "wind_clean.csv", index_col=0, parse_dates=True)
    demand = pd.read_csv(DATA_DIR / "demand_clean.csv", index_col=0, parse_dates=True)
    week_s, week_e = "2018-07-02", "2018-07-08"
    w, d = wind.loc[week_s:week_e], demand.loc[week_s:week_e]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4.5), sharex=True)
    ax1.plot(w.index, w["wind_power"], lw=0.5, color="#1565C0")
    ax1.set_ylabel("Wind Power (normalised)")
    ax1.set_ylim(0, 1.05)
    ax2.plot(d.index, d["demand"], lw=0.5, color="#C62828")
    ax2.set_ylabel("Demand (normalised)")
    ax2.set_xlabel("Date")
    fig.suptitle("Fig.2  One-week wind power and load demand profiles (July 2018)")
    save("fig2_timeseries.png")


# ═══════════════════════════════════════════════════════════════════════
# Fig 3: Best-model prediction vs actual — REAL data from experiment CSVs
# =========================================================================
def fig3_best_predictions():
    # Wind: Transformer (best model)
    w_pred = pd.read_csv(RES_DIR / "wind_transformer_predictions.csv")
    # Demand: LSTM (best model)
    d_pred = pd.read_csv(RES_DIR / "demand_lstm_predictions.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    # Plot the first 200 consecutive test samples
    n = 200
    for ax, df, label, color, ds_name in [
        (ax1, w_pred, "Wind (Transformer)", "#1565C0", "wind"),
        (ax2, d_pred, "Demand (LSTM)", "#C62828", "demand"),
    ]:
        acts = df["actual"].values[-n:]
        preds = df["prediction"].values[-n:]
        mae = np.abs(preds - acts).mean()
        ax.plot(acts, lw=1.2, color=color, label="Actual")
        ax.plot(preds, lw=0.8, color="#FF8F00", alpha=0.8, label="Predicted")
        ax.set_title(f"{label}  (MAE={mae:.4f})", fontsize=10)
        ax.set_ylabel("Normalised Power")
        ax.legend(fontsize=7)

    fig.suptitle("Fig.3  Best-model prediction vs actual (test set samples)")
    save("fig3_predictions.png")


# ═══════════════════════════════════════════════════════════════════════
# Fig 4: 4-model scatter — REAL data from experiment CSVs
# =========================================================================
def fig4_scatter():
    models = [
        ("LSTM",       RES_DIR / "wind_lstm_predictions.csv"),
        ("Transformer", RES_DIR / "wind_transformer_predictions.csv"),
        ("PatchTST",    RES_DIR / "wind_patchtst_preds.csv"),
        ("ARIMA",       RES_DIR / "wind_arima_predictions.csv"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7, 6.5))
    for ax, (name, csv_path) in zip(axes.flat, models):
        df = pd.read_csv(csv_path)
        acts = df["actual"].values
        preds = df["prediction"].values
        mae = np.abs(preds - acts).mean()
        # Downsample for cleaner scatter
        idx = np.random.choice(len(acts), min(300, len(acts)), replace=False)
        ax.scatter(acts[idx], preds[idx], s=5, alpha=0.5, color="#1565C0")
        ax.plot([acts.min(), acts.max()], [acts.min(), acts.max()], "r--", lw=0.8)
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        ax.set_title(f"{name}  (MAE={mae:.4f})", fontsize=9)

    fig.suptitle("Fig.4  Predicted vs actual scatter across four models (Wind power)")
    save("fig4_scatter.png")


# ═══════════════════════════════════════════════════════════════════════
# Fig 5: DA-CP vs Symmetric CP — REAL data from patch_results.json
# =========================================================================
def fig5_cp_comparison():
    w = patch["wind"]["cp_h2h"]
    d = patch["demand"]["cp_h2h"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3))
    for ax, cp, title in [
        (ax1, w, "Wind (LSTM, a=0.10)"),
        (ax2, d, "Demand (LSTM, a=0.10)"),
    ]:
        labels = ["Coverage", "Up Viol.", "Dn Viol."]
        sym = [cp["symmetric"]["coverage_pct"], cp["symmetric"]["up_violation_pct"], cp["symmetric"]["dn_violation_pct"]]
        asym = [cp["asymmetric"]["coverage_pct"], cp["asymmetric"]["up_violation_pct"], cp["asymmetric"]["dn_violation_pct"]]
        x = np.arange(3); wb = 0.25
        ax.bar(x - wb/2, sym, wb, label="Symmetric CP", color="#90CAF9", edgecolor="gray", lw=0.5)
        ax.bar(x + wb/2, asym, wb, label="DA-CP", color="#FF8F00", edgecolor="gray", lw=0.5)
        ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_title(title); ax.legend(fontsize=7)

    fig.suptitle("Fig.5  DA-CP vs symmetric CP: coverage and violation rates")
    save("fig5_cp_comparison.png")


# ═══════════════════════════════════════════════════════════════════════
# Fig 6: Window-length ablation — REAL data from patch_results.json
# =========================================================================
def fig6_window_ablation():
    w_abl = patch["wind"]["window_ablation"]
    d_abl = patch["demand"]["window_ablation"]
    windows = [30, 60, 90, 120]
    w_mae = [w_abl[str(x)] for x in windows]
    d_mae = [d_abl[str(x)] for x in windows]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(windows, w_mae, "o-", lw=1.5, color="#1565C0", label="Wind")
    ax.plot(windows, d_mae, "s-", lw=1.5, color="#C62828", label="Demand")
    ax.set_xlabel("Window Length (minutes)"); ax.set_ylabel("MAE")
    ax.set_xticks(windows); ax.legend()
    ax.set_title("Fig.6  Window-length ablation (LSTM, hidden=64, n_layers=2)")
    save("fig6_window_ablation.png")


# ═══════════════════════════════════════════════════════════════════════
# Fig 7: Violation rate verification — REAL data from all_experiments.json
# =========================================================================
def fig7_violation():
    cap_w = main["wind"]["capacity"]
    cap_d = main["demand"]["capacity"]
    models = ["LSTM", "Transformer", "PatchTST"]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(models)); wb = 0.3
    wv = [cap_w[m]["violation_rate_total"] for m in models]
    dv = [cap_d[m]["violation_rate_total"] for m in models]
    ax.bar(x - wb/2, wv, wb, label="Wind", color="#1565C0", edgecolor="gray", lw=0.5)
    ax.bar(x + wb/2, dv, wb, label="Demand", color="#C62828", edgecolor="gray", lw=0.5)
    ax.axhline(y=10, color="gray", ls="--", lw=1, label="a=0.10 bound")
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("Violation Rate (%)"); ax.set_ylim(0, 14); ax.legend(fontsize=7)
    ax.set_title("Fig.7  Capacity violation rate verification (DA-CP, a=0.10)")
    save("fig7_violation.png")


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating figures from REAL experiment data ...\n")
    fig1_framework()
    fig2_timeseries()
    fig3_best_predictions()
    fig4_scatter()
    fig5_cp_comparison()
    fig6_window_ablation()
    fig7_violation()
    print(f"\n7 figures saved to {FIG_DIR}/")
