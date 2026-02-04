"""XGBoost Model Training and Evaluation for Wine Classification."""

import json
import logging
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

# Constants
OUTPUT_DIR: str = "output"
MODEL_FILENAME: str = "xgb_model.pkl"
FIGURE_DPI: int = 150
RANDOM_STATE: int = 42
CV_FOLDS: int = 5
N_ITER_SEARCH: int = 20
TARGET_COLUMN: str = "target"
CLASS_NAMES: list[str] = ["class_0", "class_1", "class_2"]
PARAM_DISTRIBUTIONS: dict = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
}


def _load_train_test_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load train and test data from parquet files."""
    output_path = Path(OUTPUT_DIR)

    train_df = pl.read_parquet(output_path / "train.parquet")
    test_df = pl.read_parquet(output_path / "test.parquet")

    feature_cols = [col for col in train_df.columns if col != TARGET_COLUMN]

    X_train = train_df.select(feature_cols).to_numpy()
    y_train = train_df[TARGET_COLUMN].to_numpy()
    X_test = test_df.select(feature_cols).to_numpy()
    y_test = test_df[TARGET_COLUMN].to_numpy()

    logging.info(f"Loaded train data: X_train {X_train.shape}, y_train {y_train.shape}")
    logging.info(f"Loaded test data: X_test {X_test.shape}, y_test {y_test.shape}")

    return X_train, y_train, X_test, y_test


def _run_hyperparameter_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[XGBClassifier, dict]:
    """Run RandomizedSearchCV for hyperparameter tuning."""
    base_model = XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
        use_label_encoder=False,
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER_SEARCH,
        scoring="accuracy",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    logging.info("Starting hyperparameter tuning with RandomizedSearchCV")
    search.fit(X_train, y_train)

    logging.info(f"Best parameters: {json.dumps(search.best_params_, indent=2, default=str)}")
    logging.info(f"Best CV accuracy: {search.best_score_:.4f}")

    return search.best_estimator_, {
        "best_params": search.best_params_,
        "best_cv_accuracy": float(search.best_score_),
        "cv_results": search.cv_results_,
    }


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Calculate accuracy, precision, recall, F1-score (macro and per-class)."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }

    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    f1_per_class = f1_score(y_true, y_pred, average=None)

    metrics["per_class"] = {}
    for i, class_name in enumerate(CLASS_NAMES):
        metrics["per_class"][class_name] = {
            "precision": float(precision_per_class[i]),
            "recall": float(recall_per_class[i]),
            "f1": float(f1_per_class[i]),
        }

    logging.info(f"Evaluation metrics:\n{json.dumps(metrics, indent=2, default=str)}")
    return metrics


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    """Generate heatmap of confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    filepath = output_path / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(filepath, dpi=FIGURE_DPI)
    plt.close()
    logging.info(f"Saved confusion matrix to {filepath}")


def _plot_feature_importance(
    model: XGBClassifier,
    feature_names: list[str],
    output_path: Path,
) -> None:
    """Generate horizontal bar chart of feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.barh(range(len(indices)), importances[indices], align="center")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Feature Importance")
    ax.set_title("XGBoost Feature Importance")

    filepath = output_path / "feature_importance.png"
    plt.tight_layout()
    plt.savefig(filepath, dpi=FIGURE_DPI)
    plt.close()
    logging.info(f"Saved feature importance plot to {filepath}")


def _save_model(
    model: XGBClassifier,
    output_path: Path,
) -> None:
    """Save model using pickle."""
    filepath = output_path / MODEL_FILENAME
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    logging.info(f"Saved model to {filepath}")


def _save_tuning_results(
    tuning_results: dict,
    output_path: Path,
) -> None:
    """Save CV results to JSON."""
    # Convert numpy types for JSON serialization
    results_to_save = {
        "best_params": tuning_results["best_params"],
        "best_cv_accuracy": tuning_results["best_cv_accuracy"],
        "n_iterations": N_ITER_SEARCH,
        "cv_folds": CV_FOLDS,
    }

    filepath = output_path / "tuning_results.json"
    with open(filepath, "w") as f:
        json.dump(results_to_save, f, indent=2, default=str)
    logging.info(f"Saved tuning results to {filepath}")


def _save_evaluation_metrics(
    metrics: dict,
    output_path: Path,
) -> None:
    """Save final metrics to JSON."""
    filepath = output_path / "evaluation_metrics.json"
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logging.info(f"Saved evaluation metrics to {filepath}")


def run_model_training() -> None:
    """Main orchestration function for model training and evaluation."""
    start_time = time.time()
    logging.info("Starting XGBoost Model Training")

    output_path = Path(OUTPUT_DIR)
    X_train, y_train, X_test, y_test = _load_train_test_data()

    # Get feature names from train data
    train_df = pl.read_parquet(output_path / "train.parquet")
    feature_names = [col for col in train_df.columns if col != TARGET_COLUMN]

    # Hyperparameter tuning
    best_model, tuning_results = _run_hyperparameter_tuning(X_train, y_train)

    # Predictions and evaluation
    y_pred = best_model.predict(X_test)
    metrics = _compute_metrics(y_test, y_pred)

    # Visualizations
    _plot_confusion_matrix(y_test, y_pred, output_path)
    _plot_feature_importance(best_model, feature_names, output_path)

    # Save outputs
    _save_model(best_model, output_path)
    _save_tuning_results(tuning_results, output_path)
    _save_evaluation_metrics(metrics, output_path)

    elapsed_time = time.time() - start_time
    logging.info(f"Elapsed time: {elapsed_time:.2f} seconds")
    logging.info("Model training and evaluation completed successfully")


if __name__ == "__main__":
    run_model_training()
