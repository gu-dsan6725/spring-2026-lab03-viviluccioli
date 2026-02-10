"""Generate a comprehensive model evaluation report from trained model artifacts.

Loads a trained model, test data, and existing evaluation metrics,
then fills in a report template and saves the completed report.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

TOP_N_FEATURES = 5
DEFAULT_OUTPUT_DIR = "output"
REPORT_FILENAME = "full_report.md"


def _find_model_file(
    output_dir: Path,
) -> Path:
    """Find the first .joblib or .pkl model file in the output directory."""
    for pattern in ["*.joblib", "*.pkl"]:
        files = list(output_dir.glob(pattern))
        if files:
            logger.info(f"Found model file: {files[0]}")
            return files[0]

    raise FileNotFoundError(f"No .joblib or .pkl model file found in {output_dir}")


def _load_existing_metrics(
    output_dir: Path,
) -> dict:
    """Load metrics from evaluation_report.md if it exists."""
    report_path = output_dir / "evaluation_report.md"
    metrics = {}

    if not report_path.exists():
        logger.info("No existing evaluation_report.md found.")
        return metrics

    logger.info(f"Reading existing metrics from {report_path}")
    content = report_path.read_text()

    for line in content.splitlines():
        line = line.strip()
        if line.startswith("|") and "|" in line[1:]:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) == 2 and parts[0] not in ("Metric", "--------"):
                try:
                    value_str = parts[1].replace("%", "")
                    metrics[parts[0]] = float(value_str)
                except ValueError:
                    pass

    logger.info(f"Loaded {len(metrics)} existing metrics")
    return metrics


def _extract_model_info(
    model_path: Path,
) -> dict:
    """Load model and extract type, hyperparameters, and feature importance."""
    model = joblib.load(model_path)
    model_type = type(model).__name__

    params = model.get_params()
    key_params = {
        k: v
        for k, v in params.items()
        if v is not None
        and k
        in [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "objective",
            "random_state",
            "subsample",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
            "min_child_weight",
        ]
    }

    logger.info(f"Model type: {model_type}")
    logger.info(
        f"Hyperparameters:\n{json.dumps({k: str(v) for k, v in key_params.items()}, indent=2)}"
    )

    return {
        "model": model,
        "model_type": model_type,
        "params": key_params,
    }


def _load_test_data(
    output_dir: Path,
) -> tuple:
    """Load test features and labels from parquet files."""
    x_test = pl.read_parquet(output_dir / "x_test.parquet")
    y_test = pl.read_parquet(output_dir / "y_test.parquet")

    logger.info(f"Loaded test data: x={x_test.shape}, y={y_test.shape}")
    return x_test, y_test


def _compute_metrics(
    model,
    x_test: pl.DataFrame,
    y_test: pl.DataFrame,
    existing_metrics: dict,
) -> dict:
    """Compute prediction metrics and distribution statistics."""
    preds = model.predict(x_test.to_numpy())
    y_arr = y_test.to_numpy().flatten()
    errors = y_arr - preds

    rmse = float(np.sqrt(mean_squared_error(y_arr, preds)))
    mae = float(mean_absolute_error(y_arr, preds))
    r2 = float(r2_score(y_arr, preds))
    mape = float(np.mean(np.abs(errors / np.where(y_arr == 0, 1, y_arr))) * 100)

    metrics = {
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R-squared": round(r2, 4),
        "MAPE": f"{round(mape, 2)}%",
        "Mean prediction": round(float(np.mean(preds)), 4),
        "Prediction std": round(float(np.std(preds)), 4),
        "Mean error": round(float(np.mean(errors)), 4),
        "Error std": round(float(np.std(errors)), 4),
    }

    logger.info(f"Computed metrics:\n{json.dumps(metrics, indent=2)}")
    return metrics


def _get_feature_importance(
    model,
    feature_names: list,
    top_n: int = TOP_N_FEATURES,
) -> list:
    """Extract top N features ranked by importance."""
    importance = model.feature_importances_
    ranked = sorted(
        zip(feature_names, importance),
        key=lambda x: -x[1],
    )

    top_features = [
        {"rank": i + 1, "name": name, "score": round(float(score), 4)}
        for i, (name, score) in enumerate(ranked[:top_n])
    ]

    logger.info(f"Top {top_n} features: {top_features}")
    return top_features


def _get_dataset_info(
    output_dir: Path,
    x_test: pl.DataFrame,
) -> dict:
    """Gather dataset size information."""
    x_train = pl.read_parquet(output_dir / "x_train.parquet")

    total = x_train.shape[0] + x_test.shape[0]
    info = {
        "total": total,
        "train": x_train.shape[0],
        "test": x_test.shape[0],
        "n_features": x_test.shape[1],
        "feature_names": x_test.columns,
    }

    logger.info(
        f"Dataset: {info['total']} total, "
        f"{info['train']} train, "
        f"{info['test']} test, "
        f"{info['n_features']} features"
    )
    return info


def _build_report(
    dataset_info: dict,
    model_info: dict,
    metrics: dict,
    top_features: list,
) -> str:
    """Fill in the report template with actual data."""
    lines = []

    lines.append("# Model Evaluation Report")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    r2_val = metrics.get("R-squared", "N/A")
    rmse_val = metrics.get("RMSE", "N/A")
    mape_val = metrics.get("MAPE", "N/A")
    lines.append(
        f"An {model_info['model_type']} regression model was trained to predict "
        f"the target variable. The model achieves an R-squared of {r2_val} "
        f"with an RMSE of {rmse_val}, indicating strong predictive performance "
        f"with room for improvement where MAPE reaches {mape_val}."
    )
    lines.append("")

    # Dataset Overview
    lines.append("## Dataset Overview")
    lines.append("")
    lines.append("| Property | Value |")
    lines.append("|----------|-------|")
    lines.append(f"| Total samples | {dataset_info['total']:,} |")
    lines.append(f"| Training samples | {dataset_info['train']:,} |")
    lines.append(f"| Test samples | {dataset_info['test']:,} |")
    lines.append(f"| Number of features | {dataset_info['n_features']} |")
    lines.append("| Target variable | MedHouseVal |")
    lines.append("")

    # Model Configuration
    lines.append("## Model Configuration")
    lines.append("")
    lines.append("| Hyperparameter | Value |")
    lines.append("|----------------|-------|")
    lines.append(f"| Model type | {model_info['model_type']} |")
    for param_name, param_value in model_info["params"].items():
        lines.append(f"| {param_name} | {param_value} |")
    lines.append("")

    # Performance Metrics
    lines.append("## Performance Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for metric_name, metric_value in metrics.items():
        lines.append(f"| {metric_name} | {metric_value} |")
    lines.append("")

    # Feature Importance
    lines.append(f"## Feature Importance (Top {len(top_features)})")
    lines.append("")
    lines.append("| Rank | Feature | Importance Score |")
    lines.append("|------|---------|-----------------:|")
    for feat in top_features:
        lines.append(f"| {feat['rank']} | {feat['name']} | {feat['score']} |")
    lines.append("")

    # Recommendations
    lines.append("## Recommendations for Improvement")
    lines.append("")
    lines.append(
        "1. **Hyperparameter tuning**: Run cross-validated grid or Bayesian "
        "search over learning_rate, max_depth, n_estimators, and subsample "
        "to find a better configuration."
    )
    lines.append(
        "2. **Feature engineering**: The top features suggest room for "
        "derived features. Consider interactions, polynomial terms, or "
        "domain-specific transformations."
    )
    lines.append(
        "3. **Target transformation**: If the MAPE is high, consider "
        "log-transforming the target variable to reduce the impact of "
        "outliers and improve predictions on high-value samples."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Generate a model evaluation report from artifacts in the output dir."""
    parser = argparse.ArgumentParser(
        description="Generate a model evaluation report from trained artifacts",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        nargs="?",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory containing model artifacts (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    start_time = time.time()
    output_dir = Path(args.output_dir)

    logger.info(f"Generating report from artifacts in: {output_dir}")

    # Step 1: Read existing metrics
    existing_metrics = _load_existing_metrics(output_dir)

    # Step 2: Load model
    model_path = _find_model_file(output_dir)
    model_info = _extract_model_info(model_path)

    # Step 3: Load test data and compute metrics
    x_test, y_test = _load_test_data(output_dir)
    metrics = _compute_metrics(
        model_info["model"],
        x_test,
        y_test,
        existing_metrics,
    )

    # Step 4: Feature importance
    top_features = _get_feature_importance(
        model_info["model"],
        x_test.columns,
    )

    # Step 5: Dataset info
    dataset_info = _get_dataset_info(output_dir, x_test)

    # Step 6: Build and save report
    report_content = _build_report(
        dataset_info,
        model_info,
        metrics,
        top_features,
    )

    report_path = output_dir / REPORT_FILENAME
    report_path.write_text(report_content)
    logger.info(f"Report saved to {report_path}")

    elapsed = time.time() - start_time
    logger.info(f"Report generation completed in {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
