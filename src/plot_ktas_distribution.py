"""Plot class distribution of KTAS expert labels before and after SMOTE.

This script creates a grouped bar chart titled "Distribution of KTAS Expert Labels".
It compares raw label counts with the balanced counts produced after applying SMOTE
in the preprocessing step. The figure is saved to
`outputs/plots/ktas_distribution.png`.

Usage (from project root):
    python -m src.plot_ktas_distribution
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = PROJECT_ROOT / "data" / "data.csv"  # Raw dataset path
PROCESSED_DATA_PKL = PROJECT_ROOT / "outputs" / "processed_data.pkl"
PLOT_OUT = PROJECT_ROOT / "outputs" / "plots" / "ktas_distribution.png"


def get_raw_distribution(csv_path: Path | str) -> List[int]:
    """Return list of counts for KTAS levels 1-5 from the raw CSV file."""
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Raw data file not found: {csv_path}")

    # Attempt to read with utf-8 encoding first, fall back to latin1 with errors ignored
    try:
        df = pd.read_csv(
            csv_path,
            engine="python",
            on_bad_lines="skip",
            encoding="utf-8",
            sep=";",
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            csv_path,
            engine="python",
            on_bad_lines="skip",
            encoding="latin1",
            encoding_errors="ignore" if hasattr(pd, "options") else "ignore",
            sep=";",
        )
    if "KTAS_expert" not in df.columns:
        raise KeyError("Column 'KTAS_expert' not found in raw data")

    value_counts = df["KTAS_expert"].value_counts().sort_index()
    return [int(value_counts.get(level, 0)) for level in range(1, 6)]


def get_smote_distribution(pkl_path: Path | str) -> List[int]:
    """Return list of counts for KTAS levels 1-5 from the post-SMOTE labels."""
    if not Path(pkl_path).exists():
        raise FileNotFoundError(f"Processed data file not found: {pkl_path}")

    _, y_res = joblib.load(pkl_path)
    # y_res may be a pandas Series or NumPy array
    if hasattr(y_res, "value_counts"):
        value_counts = y_res.value_counts().sort_index()
    else:
        value_counts = pd.Series(y_res).value_counts().sort_index()

    return [int(value_counts.get(level, 0)) for level in range(1, 6)]


def plot_distributions(before: List[int], after: List[int], save_path: Path | str) -> None:
    """Generate and save the grouped bar chart comparing distributions."""
    os.makedirs(Path(save_path).parent, exist_ok=True)

    x = np.arange(1, 6)  # KTAS levels 1-5
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width / 2, before, width, label="Before SMOTE", color="#1f77b4")
    ax.bar(x + width / 2, after, width, label="After SMOTE", color="#ff7f0e")

    ax.set_xlabel("KTAS Levels (1 to 5)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of KTAS Expert Labels")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(save_path)
    print(f"✅ Saved distribution plot to {save_path}")
    # Uncomment below to show interactively when running locally
    # plt.show()


def main() -> None:
    """Main entry point."""
    print("Generating KTAS distribution plot…")
    before_counts = get_raw_distribution(DATA_CSV)
    after_counts = get_smote_distribution(PROCESSED_DATA_PKL)
    plot_distributions(before_counts, after_counts, PLOT_OUT)


if __name__ == "__main__":
    main()
