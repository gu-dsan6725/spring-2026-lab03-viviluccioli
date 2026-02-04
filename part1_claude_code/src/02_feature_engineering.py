"""Feature Engineering for Wine Classification Dataset."""

import json
import logging
import time
from pathlib import Path

import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

# Constants
OUTPUT_DIR: str = "output"
RAW_DATA_FILE: str = "wine_raw.parquet"
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42
TARGET_COLUMN: str = "target"


def _load_raw_data() -> pl.DataFrame:
    """Load raw data from parquet file."""
    filepath = Path(OUTPUT_DIR) / RAW_DATA_FILE
    df = pl.read_parquet(filepath)
    logging.info(f"Loaded raw data from {filepath} with shape: {df.shape}")
    return df


def _create_derived_features(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Add 3 new derived features."""
    df = df.with_columns(
        [
            (pl.col("alcohol") / pl.col("malic_acid")).alias("alcohol_to_malic_acid"),
            (pl.col("total_phenols") * pl.col("flavanoids")).alias(
                "total_phenols_flavanoids_interaction"
            ),
            (pl.col("color_intensity") / pl.col("hue")).alias("color_intensity_hue_ratio"),
        ]
    )
    logging.info(
        "Created 3 derived features: alcohol_to_malic_acid, "
        "total_phenols_flavanoids_interaction, color_intensity_hue_ratio"
    )
    return df


def _handle_infinite_values(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Replace inf/nan with column medians."""
    feature_cols = [col for col in df.columns if col != TARGET_COLUMN]

    for col in feature_cols:
        median_val = df.filter(pl.col(col).is_finite())[col].median()
        df = df.with_columns(
            pl.when(pl.col(col).is_finite()).then(pl.col(col)).otherwise(median_val).alias(col)
        )

    logging.info("Handled infinite and NaN values by replacing with column medians")
    return df


def _log_feature_statistics(
    df: pl.DataFrame,
    stage: str,
) -> None:
    """Log mean/std for all features."""
    feature_cols = [col for col in df.columns if col != TARGET_COLUMN]
    stats = {}

    for col in feature_cols:
        stats[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
        }

    logging.info(f"Feature statistics ({stage}):\n{json.dumps(stats, indent=2, default=str)}")


def _scale_features(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, StandardScaler]:
    """Apply StandardScaler to all numeric features (excluding target)."""
    feature_cols = [col for col in df.columns if col != TARGET_COLUMN]
    target = df[TARGET_COLUMN]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.select(feature_cols).to_numpy())

    scaled_df = pl.DataFrame(scaled_data, schema=feature_cols)
    scaled_df = scaled_df.with_columns(target.alias(TARGET_COLUMN))

    logging.info(f"Scaled {len(feature_cols)} features using StandardScaler")
    return scaled_df, scaler


def _stratified_split(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split data using stratified sampling."""
    feature_cols = [col for col in df.columns if col != TARGET_COLUMN]
    X = df.select(feature_cols).to_numpy()
    y = df[TARGET_COLUMN].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_df = pl.DataFrame(X_train, schema=feature_cols)
    train_df = train_df.with_columns(pl.Series(TARGET_COLUMN, y_train))

    test_df = pl.DataFrame(X_test, schema=feature_cols)
    test_df = test_df.with_columns(pl.Series(TARGET_COLUMN, y_test))

    logging.info(f"Split data: train shape {train_df.shape}, test shape {test_df.shape}")
    return train_df, test_df


def _save_splits(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
) -> None:
    """Save train and test DataFrames as parquet files."""
    output_path = Path(OUTPUT_DIR)

    train_filepath = output_path / "train.parquet"
    test_filepath = output_path / "test.parquet"

    train_df.write_parquet(train_filepath)
    test_df.write_parquet(test_filepath)

    logging.info(f"Saved train data to {train_filepath}")
    logging.info(f"Saved test data to {test_filepath}")


def run_feature_engineering() -> None:
    """Main orchestration function for feature engineering."""
    start_time = time.time()
    logging.info("Starting Feature Engineering")

    df = _load_raw_data()
    df = _create_derived_features(df)
    df = _handle_infinite_values(df)

    _log_feature_statistics(df, "before scaling")

    scaled_df, _ = _scale_features(df)

    _log_feature_statistics(scaled_df, "after scaling")

    train_df, test_df = _stratified_split(scaled_df)
    _save_splits(train_df, test_df)

    elapsed_time = time.time() - start_time
    logging.info(f"Elapsed time: {elapsed_time:.2f} seconds")
    logging.info("Feature Engineering completed successfully")


if __name__ == "__main__":
    run_feature_engineering()
