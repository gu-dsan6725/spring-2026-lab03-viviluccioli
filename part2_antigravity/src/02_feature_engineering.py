import logging
import time
from pathlib import Path

import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

# Constants
OUTPUT_DIR = Path(__file__).parent.parent / "output"
RAW_DATA_FILE = OUTPUT_DIR / "wine_raw.parquet"
TRAIN_DATA_FILE = OUTPUT_DIR / "train.parquet"
TEST_DATA_FILE = OUTPUT_DIR / "test.parquet"


def _load_raw_data() -> pl.DataFrame:
    """Load raw data from parquet."""
    logging.info(f"Loading raw data from {RAW_DATA_FILE}...")
    return pl.read_parquet(RAW_DATA_FILE)


def _create_derived_features(df: pl.DataFrame) -> pl.DataFrame:
    """Create derived features."""
    logging.info("Creating derived features...")

    # Check if columns exist
    required_cols = [
        "alcohol",
        "malic_acid",
        "total_phenols",
        "flavanoids",
        "color_intensity",
        "hue",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for feature engineering: {missing_cols}")

    df_derived = df.with_columns(
        [
            (pl.col("alcohol") / pl.col("malic_acid")).alias("alcohol_to_malic_acid"),
            (pl.col("total_phenols") * pl.col("flavanoids")).alias(
                "total_phenols_flavanoids_interaction"
            ),
            (pl.col("color_intensity") / pl.col("hue")).alias("color_intensity_hue_ratio"),
        ]
    )

    logging.info(f"Derived features created. New shape: {df_derived.shape}")
    return df_derived


def _scale_features(df: pl.DataFrame, target_col: str = "target") -> pl.DataFrame:
    """Apply StandardScaler to all numeric features."""
    logging.info("Scaling numeric features...")

    features = [c for c in df.columns if c != target_col]

    # Log stats before scaling
    logging.info("Stats before scaling (first 3 features):")
    logging.info(df.select(features[:3]).describe())

    scaler = StandardScaler()
    # Convert to pandas/numpy for sklearn scaling
    X = df.select(features).to_numpy()
    X_scaled = scaler.fit_transform(X)

    # Create new DataFrame with scaled values
    df_scaled = pl.DataFrame(X_scaled, schema=features)

    # Attach target back
    df_final = df_scaled.with_columns(df[target_col])

    # Log stats after scaling
    logging.info("Stats after scaling (first 3 features):")
    logging.info(df_final.select(features[:3]).describe())

    return df_final


def _split_and_save_data(df: pl.DataFrame, target_col: str = "target") -> None:
    """Perform stratified train/test split and save."""
    logging.info("Splitting data into train and test sets...")

    # Polars doesn't have a direct train_test_split that is stratified easily without helper,
    # so we convert to pandas/numpy for splitting or use indexes.
    # To keep it simple and safe with sklearn split:
    df_pd = df.to_pandas()
    X = df_pd.drop(columns=[target_col])
    y = df_pd[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Convert back to Polars
    train_df = pl.from_pandas(X_train).with_columns(pl.Series(target_col, y_train))
    test_df = pl.from_pandas(X_test).with_columns(pl.Series(target_col, y_test))

    logging.info(f"Train set shape: {train_df.shape}")
    logging.info(f"Test set shape: {test_df.shape}")

    logging.info(f"Saving train data to {TRAIN_DATA_FILE}...")
    train_df.write_parquet(TRAIN_DATA_FILE)

    logging.info(f"Saving test data to {TEST_DATA_FILE}...")
    test_df.write_parquet(TEST_DATA_FILE)
    logging.info("Data splitting and saving complete.")


def main() -> None:
    """Main execution function."""
    start_time = time.time()
    logging.info("Starting Feature Engineering Pipeline Step 2/3")

    df = _load_raw_data()
    df_derived = _create_derived_features(df)
    df_scaled = _scale_features(df_derived)
    _split_and_save_data(df_scaled)

    elapsed_time = time.time() - start_time
    logging.info(f"Feature Engineering Pipeline completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
