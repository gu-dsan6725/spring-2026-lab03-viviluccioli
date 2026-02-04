"""Exploratory Data Analysis for Wine Classification Dataset."""

import json
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.datasets import load_wine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

# Constants
OUTPUT_DIR: str = "output"
FIGURE_DPI: int = 150
IQR_MULTIPLIER: float = 1.5


def _ensure_output_dir() -> Path:
    """Create output directory if it doesn't exist."""
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory ensured: {output_path}")
    return output_path


def _load_wine_dataset() -> pl.DataFrame:
    """Load Wine dataset from sklearn and convert to polars DataFrame."""
    wine = load_wine()
    df = pl.DataFrame(wine.data, schema=wine.feature_names)
    df = df.with_columns(pl.Series("target", wine.target))
    logging.info(f"Loaded Wine dataset with shape: {df.shape}")
    return df


def _compute_summary_statistics(
    df: pl.DataFrame,
) -> dict:
    """Calculate summary statistics for all features."""
    feature_cols = [col for col in df.columns if col != "target"]
    stats = {}

    for col in feature_cols:
        col_data = df[col]
        stats[col] = {
            "mean": float(col_data.mean()),
            "std": float(col_data.std()),
            "min": float(col_data.min()),
            "max": float(col_data.max()),
            "q1": float(col_data.quantile(0.25)),
            "q2_median": float(col_data.quantile(0.50)),
            "q3": float(col_data.quantile(0.75)),
        }

    logging.info(f"Summary statistics:\n{json.dumps(stats, indent=2, default=str)}")
    return stats


def _check_class_balance(
    df: pl.DataFrame,
) -> dict:
    """Count samples per wine class and log distribution."""
    class_counts = df.group_by("target").len().sort("target")
    balance = {f"class_{row['target']}": row["len"] for row in class_counts.iter_rows(named=True)}
    logging.info(f"Class balance:\n{json.dumps(balance, indent=2, default=str)}")
    return balance


def _detect_outliers_iqr(
    df: pl.DataFrame,
) -> dict:
    """Use IQR method to identify outliers per feature."""
    feature_cols = [col for col in df.columns if col != "target"]
    outliers = {}

    for col in feature_cols:
        col_data = df[col]
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - IQR_MULTIPLIER * iqr
        upper_bound = q3 + IQR_MULTIPLIER * iqr

        outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
        outlier_count = outlier_mask.sum()
        outliers[col] = {
            "count": int(outlier_count),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
        }

    logging.info(f"Outlier detection (IQR method):\n{json.dumps(outliers, indent=2, default=str)}")
    return outliers


def _plot_distributions(
    df: pl.DataFrame,
    output_path: Path,
) -> None:
    """Generate 4x4 subplot grid of histograms for all 13 features."""
    feature_cols = [col for col in df.columns if col != "target"]
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        axes[i].hist(df[col].to_numpy(), bins=20, edgecolor="black", alpha=0.7)
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Frequency")

    # Hide unused subplots
    for j in range(len(feature_cols), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    filepath = output_path / "distributions.png"
    plt.savefig(filepath, dpi=FIGURE_DPI)
    plt.close()
    logging.info(f"Saved distributions plot to {filepath}")


def _plot_correlation_matrix(
    df: pl.DataFrame,
    output_path: Path,
) -> None:
    """Create seaborn heatmap of feature correlations."""
    feature_cols = [col for col in df.columns if col != "target"]
    corr_matrix = df.select(feature_cols).to_pandas().corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
        annot_kws={"size": 8},
    )
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()

    filepath = output_path / "correlation_matrix.png"
    plt.savefig(filepath, dpi=FIGURE_DPI)
    plt.close()
    logging.info(f"Saved correlation matrix to {filepath}")


def _save_raw_data(
    df: pl.DataFrame,
    output_path: Path,
) -> None:
    """Save DataFrame as parquet."""
    filepath = output_path / "wine_raw.parquet"
    df.write_parquet(filepath)
    logging.info(f"Saved raw data to {filepath}")


def run_eda() -> None:
    """Main orchestration function for EDA."""
    start_time = time.time()
    logging.info("Starting Exploratory Data Analysis")

    output_path = _ensure_output_dir()
    df = _load_wine_dataset()

    _compute_summary_statistics(df)
    _check_class_balance(df)
    _detect_outliers_iqr(df)
    _plot_distributions(df, output_path)
    _plot_correlation_matrix(df, output_path)
    _save_raw_data(df, output_path)

    elapsed_time = time.time() - start_time
    logging.info(f"Elapsed time: {elapsed_time:.2f} seconds")
    logging.info("EDA completed successfully")


if __name__ == "__main__":
    run_eda()
