import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.datasets import load_wine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

# Constants
OUTPUT_DIR = Path(__file__).parent.parent / "output"
DISTRIBUTIONS_PLOT = OUTPUT_DIR / "distributions.png"
CORRELATION_PLOT = OUTPUT_DIR / "correlation_matrix.png"
RAW_DATA_FILE = OUTPUT_DIR / "wine_raw.parquet"


def _ensure_output_dir() -> None:
    """Ensure the output directory exists."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_data() -> pl.DataFrame:
    """Load Wine dataset and convert to Polars DataFrame."""
    logging.info("Loading Wine dataset...")
    wine = load_wine()
    data = pl.DataFrame(wine.data, schema=wine.feature_names)
    target = pl.Series("target", wine.target)

    # Add target column
    df = data.with_columns(target)
    logging.info(f"Dataset loaded. Shape: {df.shape}")
    return df


def _compute_summary_stats(df: pl.DataFrame) -> None:
    """Compute and log summary statistics for all features."""
    logging.info("Computing summary statistics...")
    stats = df.describe()
    logging.info(f"Summary Statistics:\n{stats}")

    # Specific stats logging as requested
    numeric_cols = df.columns
    if "target" in numeric_cols:
        numeric_cols.remove("target")

    for col in numeric_cols:
        mean_val = df[col].mean()
        std_val = df[col].std()
        min_val = df[col].min()
        max_val = df[col].max()
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        logging.info(
            f"Feature '{col}': Mean={mean_val:.2f}, Std={std_val:.2f}, "
            f"Min={min_val:.2f}, Max={max_val:.2f}, Q1={q1:.2f}, Q3={q3:.2f}"
        )


def _plot_distributions(df: pl.DataFrame) -> None:
    """Generate and save distribution histograms."""
    logging.info("Generating distribution plots...")
    features = [c for c in df.columns if c != "target"]
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 5 * n_rows))
    for i, feature in enumerate(features, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(df[feature], kde=True)
        plt.title(f"Distribution of {feature}")

    plt.tight_layout()
    plt.savefig(DISTRIBUTIONS_PLOT)
    plt.close()
    logging.info(f"Distributions plot saved to {DISTRIBUTIONS_PLOT}")


def _plot_correlation(df: pl.DataFrame) -> None:
    """Generate and save correlation heatmap."""
    logging.info("Generating correlation heatmap...")
    # Convert to pandas for correlation and plotting to ensure compatibility with seaborn
    corr_pd = df.to_pandas().corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_pd, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.savefig(CORRELATION_PLOT)
    plt.close()
    logging.info(f"Correlation plot saved to {CORRELATION_PLOT}")


def _check_class_balance(df: pl.DataFrame) -> None:
    """Check and log class balance."""
    logging.info("Checking class balance...")
    balance = df.group_by("target").len().sort("target")
    logging.info(f"Class Balance:\n{balance}")


def _detect_outliers(df: pl.DataFrame) -> None:
    """Detect outliers using IQR method and log counts."""
    logging.info("Detecting outliers...")
    features = [c for c in df.columns if c != "target"]

    outlier_counts = {}

    for col in features:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = df.filter((pl.col(col) < lower_bound) | (pl.col(col) > upper_bound))
        count = outliers.height
        outlier_counts[col] = count

        if count > 0:
            logging.info(f"Feature '{col}' has {count} outliers")

    logging.info(f"Outlier detection complete. Detailed counts: {outlier_counts}")


def _save_data(df: pl.DataFrame) -> None:
    """Save raw data to parquet."""
    logging.info(f"Saving data to {RAW_DATA_FILE}...")
    df.write_parquet(RAW_DATA_FILE)
    logging.info("Data saved successfully.")


def main() -> None:
    """Main execution function."""
    start_time = time.time()
    logging.info("Starting EDA Pipeline Step 1/3")

    _ensure_output_dir()

    df = _load_data()
    _compute_summary_stats(df)
    _plot_distributions(df)
    _plot_correlation(df)
    _check_class_balance(df)
    _detect_outliers(df)
    _save_data(df)

    elapsed_time = time.time() - start_time
    logging.info(f"EDA Pipeline completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
