import json
import logging
import time
from pathlib import Path
from typing import Any, List

import joblib
import matplotlib.pyplot as plt
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

# Constants
OUTPUT_DIR = Path(__file__).parent.parent / "output"
TRAIN_DATA_FILE = OUTPUT_DIR / "train.parquet"
TEST_DATA_FILE = OUTPUT_DIR / "test.parquet"
MODEL_FILE = OUTPUT_DIR / "xgb_model.pkl"
CONFUSION_MATRIX_PLOT = OUTPUT_DIR / "confusion_matrix.png"
FEATURE_IMPORTANCE_PLOT = OUTPUT_DIR / "feature_importance.png"
TUNING_RESULTS_FILE = OUTPUT_DIR / "tuning_results.json"
EVALUATION_METRICS_FILE = OUTPUT_DIR / "evaluation_metrics.json"


def _load_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load train and test data."""
    logging.info("Loading train/test data...")
    train_df = pl.read_parquet(TRAIN_DATA_FILE)
    test_df = pl.read_parquet(TEST_DATA_FILE)
    logging.info(f"Loaded train shape: {train_df.shape}, test shape: {test_df.shape}")
    return train_df, test_df


def _prepare_xy(df: pl.DataFrame, target_col: str = "target") -> tuple[Any, Any]:
    """Separate features and target, converting to numpy/pandas format."""
    # Polars to pandas/numpy for sklearn/xgboost
    df_pd = df.to_pandas()
    X = df_pd.drop(columns=[target_col])
    y = df_pd[target_col]
    return X, y


def _train_and_tune(X_train: Any, y_train: Any) -> RandomizedSearchCV:
    """Train XGBClassifier with RandomizedSearchCV."""
    logging.info("Starting hyperparameter tuning...")

    xgb = XGBClassifier(random_state=42, eval_metric="mlogloss", use_label_encoder=False)

    param_dist = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
    }

    # 5-fold stratified cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=20,
        scoring="accuracy",
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)
    logging.info(f"Best params: {search.best_params_}")
    logging.info(f"Best CV score: {search.best_score_:.4f}")

    return search


def _save_tuning_results(search: RandomizedSearchCV) -> None:
    """Save tuning results to JSON."""
    logging.info(f"Saving tuning results to {TUNING_RESULTS_FILE}...")
    results = {
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "cv_results_mean_test_score": search.cv_results_["mean_test_score"].tolist(),
    }
    with open(TUNING_RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def _evaluate_model(model: Any, X_test: Any, y_test: Any, class_names: List[str]) -> None:
    """Evaluate model and save metrics/plots."""
    logging.info("Evaluating model on test set...")

    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro")),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
    }

    logging.info(f"Evaluation Metrics:\n{json.dumps(metrics, indent=2)}")

    # Save metrics
    with open(EVALUATION_METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(CONFUSION_MATRIX_PLOT)
    plt.close()
    logging.info(f"Confusion matrix saved to {CONFUSION_MATRIX_PLOT}")


def _plot_feature_importance(model: Any, feature_names: List[str]) -> None:
    """Generate and save feature importance plot."""
    logging.info("Generating feature importance plot...")

    importances = model.feature_importances_
    indices = importances.argsort()

    plt.figure(figsize=(10, 8))
    plt.title("Feature Importance")
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PLOT)
    plt.close()
    logging.info(f"Feature importance saved to {FEATURE_IMPORTANCE_PLOT}")


def _save_model(model: Any) -> None:
    """Save the best model to pickle."""
    logging.info(f"Saving model to {MODEL_FILE}...")
    joblib.dump(model, MODEL_FILE)
    logging.info("Model saved.")


def main() -> None:
    """Main execution function."""
    start_time = time.time()
    logging.info("Starting XGBoost Model Pipeline Step 3/3")

    # Load data
    train_df, test_df = _load_data()
    X_train, y_train = _prepare_xy(train_df)
    X_test, y_test = _prepare_xy(test_df)

    # Tuning
    search = _train_and_tune(X_train, y_train)
    _save_tuning_results(search)

    best_model = search.best_estimator_

    # Evaluation
    # Class names for wine dataset are essentially 0, 1, 2 (or class_0, class_1, class_2)
    class_names = ["class_0", "class_1", "class_2"]
    _evaluate_model(best_model, X_test, y_test, class_names)

    # Feature Importance
    feature_names = X_train.columns.tolist()
    _plot_feature_importance(best_model, feature_names)

    # Save Model
    _save_model(best_model)

    elapsed_time = time.time() - start_time
    logging.info(f"Model Pipeline completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
