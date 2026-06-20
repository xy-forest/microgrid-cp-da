# 微电网功率预测与方向感知保形预测可调能力估计

Direction-Aware Conformal Prediction (DA-CP) for microgrid power forecasting and adjustable capacity estimation.

## 项目简介

本项目实现了课程论文《基于 Transformer-LSTM 与方向感知保形预测的微电网功率预测及可调能力估计》中的完整实验代码。方法流水线包含三个阶段：

1. **点预测** — LSTM、Transformer、PatchTST 和 ARIMA 模型预测未来 10 分钟的风电出力和负荷需求。
2. **不确定性量化** — 保形预测（Split CP、DA-CP、加权 CP、归一化 CP）生成具有有限样本覆盖率保证的预测区间。
3. **可调能力估计** — 将 CP 预测区间转化为具有安全边界的上调/下调可调容量。

**核心创新**：提出方向感知保形预测（DA-CP），针对微电网运行中上调不足（供电缺口）代价远大于下调过度（弃风弃电）的物理非对称性，在上调和下调方向分别采用 α/2 和 α 的差异化覆盖率目标，将上调越界率降低约 44%。

## 目录结构

```
microgrid_cp/
├── README.md
├── requirements.txt
├── src/
│   ├── data/          # 数据清洗与预处理
│   ├── models/        # LSTM、Transformer、PatchTST
│   ├── cp/            # Split CP、DA-CP、加权 CP、归一化 CP
│   ├── capacity/      # 可调能力估计与越界率验证
│   ├── experiments/   # 实验流水线
│   └── utils/         # 评估指标与工具函数
├── scripts/           # 一键运行入口与图表生成
├── data/              # 原始数据与清洗后数据
└── results/           # 实验结果输出
```

## 环境配置

```bash
pip install -r requirements.txt
```

Python 3.10+，已在 macOS (MPS) 和 Windows (CUDA) 上测试。

## 数据

Remote Microgrid Dataset（chandar-lab, Mila 实验室），偏远风-柴-储微电网 2018 年全年 1 分钟分辨率的风电出力和负荷需求数据，共 525,600 条时间戳。

## 运行

```bash
# 1. 数据清洗
python -m src.data.preprocessing

# 2. 运行全部实验（模型对比 + CP 评估 + 消融 + 可调能力验证）
python scripts/run_full_pipeline.py
```

## 主要结果

| 数据集 | 最优模型 | MAE | 越界率 (α=0.10) |
|--------|---------|-----|-----------------|
| 风电   | Transformer | 0.0322 | 7.88% |
| 负荷   | LSTM        | 0.0558 | 8.42% |

DA-CP 相比对称 CP：覆盖率提升 1.3-3.2 个百分点，上调越界率降低约 44%。全部模型的可调能力越界率均 ≤ α = 10%。

## 引用

```bibtex
@misc{microgrid-cp-da-2026,
  title   = {Transformer-LSTM with Direction-Aware Conformal Prediction for
             Microgrid Power Forecasting and Adjustable Capacity Estimation},
  author  = {[Author]},
  year    = {2026},
  url     = {https://github.com/xy-forest/microgrid-cp-da}
}
```

## 许可证

MIT

---

# Microgrid Power Forecasting with Direction-Aware Conformal Prediction

Direction-Aware Conformal Prediction (DA-CP) for microgrid power forecasting and adjustable capacity estimation.

## Overview

This repository implements the full experimental pipeline from the course paper *"Transformer-LSTM with Direction-Aware Conformal Prediction for Microgrid Power Forecasting and Adjustable Capacity Estimation"*.

The pipeline has three stages:

1. **Point forecasting** — LSTM, Transformer, PatchTST, and ARIMA models predict wind power and load demand 10 minutes ahead.
2. **Uncertainty quantification** — Conformal Prediction (Split CP, DA-CP, Weighted CP, Normalized CP) produces prediction intervals with finite-sample coverage guarantees.
3. **Adjustable capacity** — The CP intervals are translated into upward and downward adjustable capacity with provable safety bounds.

**Core contribution**: DA-CP applies differentiated coverage targets (α/2 for upward, α for downward) to reflect the asymmetric physical risk in microgrid operation — upward shortfall (power shortage) is far costlier than downward excess (curtailment). DA-CP reduces upward violation by ~44% compared to symmetric CP.

## Directory Structure

```
microgrid_cp/
├── README.md
├── requirements.txt
├── src/
│   ├── data/          # Data cleaning and preprocessing
│   ├── models/        # LSTM, Transformer, PatchTST
│   ├── cp/            # Split CP, DA-CP, Weighted CP, Normalized CP
│   ├── capacity/      # Adjustable capacity estimation
│   ├── experiments/   # Experiment pipeline
│   └── utils/         # Metrics and utilities
├── scripts/           # Entry-point scripts and figure generation
├── data/              # Raw and processed data
└── results/           # Experiment outputs
```

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.10+ on macOS (MPS) and Windows (CUDA).

## Data

The Remote Microgrid Dataset (chandar-lab, Mila) contains one year (2018) of 1-minute resolution wind power and load demand readings from a remote wind-diesel-storage microgrid.

## Running

```bash
# 1. Preprocess raw data
python -m src.data.preprocessing

# 2. Run all experiments (model comparison + CP evaluation + ablation + capacity verification)
python scripts/run_full_pipeline.py
```

## Key Results

| Dataset | Best Model | MAE | Violation Rate (α=0.10) |
|---------|-----------|-----|--------------------------|
| Wind    | Transformer | 0.0322 | 7.88% |
| Demand  | LSTM        | 0.0558 | 8.42% |

DA-CP vs symmetric CP: coverage improved by 1.3–3.2 pp, upward violation reduced by ~44%. Capacity violation rates across all models remain ≤ α = 10%.

## Citation

```bibtex
@misc{microgrid-cp-da-2026,
  title   = {Transformer-LSTM with Direction-Aware Conformal Prediction for
             Microgrid Power Forecasting and Adjustable Capacity Estimation},
  author  = {[Author]},
  year    = {2026},
  url     = {https://github.com/xy-forest/microgrid-cp-da}
}
```

## License

MIT
