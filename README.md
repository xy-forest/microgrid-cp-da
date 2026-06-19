# Microgrid Power Forecasting and Adjustable Capacity Estimation

Direction-Aware Conformal Prediction (DA-CP) for microgrid wind and load forecasting.

## Overview

This repository implements the full experimental pipeline described in the
course paper *"Transformer-LSTM with Direction-Aware Conformal Prediction for
Microgrid Power Forecasting and Adjustable Capacity Estimation"*.

The pipeline has three stages:
1. **Point forecasting** — LSTM, Transformer, PatchTST, and ARIMA models
   predict wind power and load demand 10 minutes ahead.
2. **Uncertainty quantification** — Conformal Prediction (Split CP,
   DA-CP, Weighted CP, Normalized CP) produces prediction intervals with
   finite-sample coverage guarantees.
3. **Adjustable capacity** — The CP intervals are translated into upward
   and downward adjustable capacity with provable safety bounds.

## Directory structure

```
microgrid_cp/
├── README.md
├── requirements.txt
├── src/
│   ├── data/          # Preprocessing and feature engineering
│   ├── models/        # LSTM, Transformer, PatchTST
│   ├── cp/            # Split CP, DA-CP, Weighted CP, Normalized CP
│   ├── capacity/      # Adjustable capacity estimation
│   ├── experiments/   # Experiment runner
│   └── utils/         # Metrics and helpers
├── scripts/           # Entry-point scripts
├── data/              # Raw and processed data (not tracked)
├── results/           # Experiment outputs
└── figures/           # Generated figures
```

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.10+ on macOS (MPS) and Windows (CUDA).

## Data

The Remote Microgrid Dataset (chandar-lab, Mila) contains one year (2018)
of 1-minute resolution wind power and load demand readings from a
remote wind-diesel-storage microgrid.

## Running

```bash
# 1. Preprocess raw data
python -m src.data.preprocessing

# 2. Run all experiments (models + CP + ablation + capacity)
python scripts/run_full_pipeline.py
```

## Key results

| Dataset | Best model | MAE | Violation rate (α=0.10) |
|---------|-----------|-----|--------------------------|
| Wind   | Transformer | 0.0322 | 7.88% |
| Demand | LSTM        | 0.0558 | 8.42% |

DA-CP reduces upward violation by ~44% compared to symmetric CP while
maintaining valid coverage.

## Reference

If you use this code, please cite:
```
[Author names]. Transformer-LSTM with Direction-Aware Conformal Prediction
for Microgrid Power Forecasting and Adjustable Capacity Estimation. 2026.
```

## License

MIT
