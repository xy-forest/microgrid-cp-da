#!/usr/bin/env python3
"""Entry point: run the complete microgrid CP experiment pipeline.

Usage:
    python scripts/run_full_pipeline.py

Requires cleaned data files (wind_clean.csv, demand_clean.csv) in data/processed/.
"""

import sys
from pathlib import Path

# Allow running from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.runner import run_all_experiments


def main():
    data_dir = PROJECT_ROOT / "data" / "processed"
    results_dir = PROJECT_ROOT / "results"

    if not (data_dir / "wind_clean.csv").exists():
        print("Cleaned data not found. Run src/data/preprocessing.py first.")
        sys.exit(1)

    results = run_all_experiments(data_dir, results_dir)

    # Print summary
    for ds in ["wind", "demand"]:
        models = results[ds]["models"]
        best_model = min(models, key=lambda m: models[m]["metrics"]["mae"])
        print(
            f"\n[{ds}] Best model: {best_model} "
            f"(MAE={models[best_model]['metrics']['mae']:.4f})"
        )
        cap = results[ds]["capacity"]["LSTM"]
        print(
            f"[{ds}] Violation rate: {cap['violation_rate']:.2f}% "
            f"(alpha=0.10, target ≤10%)"
        )


if __name__ == "__main__":
    main()
