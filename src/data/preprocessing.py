"""Data preprocessing pipeline for microgrid time series.

Handles: duplicate removal, missing value imputation, outlier detection,
and time alignment between wind and demand signals.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_raw_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load wind power and demand CSV files in their raw format.

    Parameters
    ----------
    data_dir : Path
        Directory containing wind_power.csv and norm_demand.csv.

    Returns
    -------
    wind, demand : pd.DataFrame
        DataFrames indexed by DateTime.
    """
    wind = pd.read_csv(data_dir / "wind_power.csv", parse_dates=["DateTime"])
    wind = wind.set_index("DateTime").sort_index()

    demand = pd.read_csv(data_dir / "norm_demand.csv")
    demand["DateTime"] = pd.to_datetime(
        {
            "year": 2018,
            "month": demand["Month"],
            "day": demand["Day"],
            "hour": demand["Hour"],
            "minute": demand["Minute"],
        }
    )
    demand = demand[["DateTime", "Load"]].set_index("DateTime").sort_index()

    return wind, demand


def remove_duplicates(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Keep only the first occurrence of each timestamp."""
    n_dups = df.index.duplicated().sum()
    if n_dups > 0:
        print(f"  [{label}] removed {n_dups} duplicate timestamps")
    return df[~df.index.duplicated(keep="first")]


def fill_missing_timestamps(
    df: pd.DataFrame, freq: str = "1T"
) -> tuple[pd.DataFrame, int]:
    """Reindex to a complete date range and forward-fill gaps.

    Parameters
    ----------
    df : pd.DataFrame
    freq : str
        Pandas frequency string. Default 1-minute.

    Returns
    -------
    df_filled, n_missing : tuple
    """
    full_range = pd.date_range(
        start=df.index.min(), end=df.index.max(), freq=freq
    )
    n_missing = len(full_range) - len(df)
    if n_missing > 0:
        df = df.reindex(full_range).ffill()
        df.index.name = "DateTime"
    return df, n_missing


def flag_outliers(
    series: pd.Series, method: str = "iqr", multiplier: float = 3.0
) -> dict:
    """Identify outliers using IQR or fixed-range methods.

    Parameters
    ----------
    series : pd.Series
    method : str
        'iqr' for inter-quartile range, 'range' for [0,1] fixed bounds.
    multiplier : float
        IQR multiplier.

    Returns
    -------
    dict with outlier_count, lower_bound, upper_bound.
    """
    if method == "range":
        n_out = ((series < 0) | (series > 1)).sum()
        return {"count": int(n_out), "lower": 0.0, "upper": 1.0}
    else:
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - multiplier * iqr, q3 + multiplier * iqr
        n_out = ((series < lo) | (series > hi)).sum()
        return {"count": int(n_out), "lower": round(lo, 4), "upper": round(hi, 4)}


def align_time_ranges(
    a: pd.DataFrame, b: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Truncate both DataFrames to their overlapping time range."""
    start = max(a.index.min(), b.index.min())
    end = min(a.index.max(), b.index.max())
    return a.loc[start:end], b.loc[start:end]


def run_preprocessing(
    data_dir: Path, output_dir: Path
) -> dict:
    """Execute the full preprocessing pipeline and save cleaned outputs.

    Parameters
    ----------
    data_dir : Path
        Directory with raw wind_power.csv and norm_demand.csv.
    output_dir : Path
        Directory for wind_clean.csv, demand_clean.csv, and cleaning_report.json.

    Returns
    -------
    dict with cleaning statistics.
    """
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"wind": {}, "demand": {}, "alignment": {}}

    wind, demand = load_raw_data(data_dir)
    report["wind"]["raw_rows"] = len(wind)
    report["demand"]["raw_rows"] = len(demand)

    # Deduplicate
    wind = remove_duplicates(wind, "wind")
    demand = remove_duplicates(demand, "demand")

    # Fill missing timestamps
    wind, miss_w = fill_missing_timestamps(wind)
    demand, miss_d = fill_missing_timestamps(demand)
    report["wind"]["missing_timestamps"] = miss_w
    report["demand"]["missing_timestamps"] = miss_d

    # Fill NaN values
    for col, df, label in [
        ("AvailableWindPower", wind, "wind"),
        ("Load", demand, "demand"),
    ]:
        n_nan = int(df[col].isna().sum())
        df[col] = df[col].ffill().bfill()
        report[label]["nan_values"] = n_nan

    # Outlier detection
    out_w = flag_outliers(wind["AvailableWindPower"], method="range")
    wind["AvailableWindPower"] = wind["AvailableWindPower"].clip(0.0, 1.0)
    report["wind"]["outliers"] = out_w

    out_d = flag_outliers(demand["Load"], method="iqr", multiplier=3.0)
    report["demand"]["outliers_iqr"] = out_d

    # Time alignment
    wind, demand = align_time_ranges(wind, demand)
    report["alignment"]["common_start"] = str(wind.index.min())
    report["alignment"]["common_end"] = str(wind.index.max())
    report["alignment"]["wind_rows"] = len(wind)
    report["alignment"]["demand_rows"] = len(demand)

    # Rename columns and save
    wind_out = wind.rename(columns={"AvailableWindPower": "wind_power"})
    demand_out = demand.rename(columns={"Load": "demand"})
    wind_out.to_csv(output_dir / "wind_clean.csv")
    demand_out.to_csv(output_dir / "demand_clean.csv")

    # Summary
    span_days = (wind.index.max() - wind.index.min()).total_seconds() / 86400.0
    report["summary"] = {
        "wind_rows": len(wind_out),
        "demand_rows": len(demand_out),
        "wind_mean": round(float(wind_out["wind_power"].mean()), 4),
        "wind_std": round(float(wind_out["wind_power"].std()), 4),
        "demand_mean": round(float(demand_out["demand"].mean()), 4),
        "demand_std": round(float(demand_out["demand"].std()), 4),
        "time_span_days": round(span_days, 1),
    }

    (output_dir / "cleaning_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )

    print(f"Cleaning complete: {len(wind_out)} wind, {len(demand_out)} demand rows")
    return report
