# Plan: Wine Classification Pipeline

## Objective

Build a complete ML classification pipeline for the UCI Wine dataset (3 classes) consisting of 3 Python scripts for EDA, feature engineering, and XGBoost model training with hyperparameter tuning.

## Steps

### Step 1: Create Project Structure

- **Files**: `part1_claude_code/src/` directory, `output/` directory
- **Action**: Create the `src/` directory inside `part1_claude_code/` and an `output/` directory at project root
- **Dependencies**: None

### Step 2: Create `01_eda.py` - Exploratory Data Analysis

- **File**: `part1_claude_code/src/01_eda.py`
- **Action**: Implement EDA script with the following functions:
  - `_ensure_output_dir()` - Create output directory
  - `_load_wine_dataset()` - Load Wine dataset from sklearn and convert to polars DataFrame with target column
  - `_compute_summary_statistics()` - Calculate mean, std, min, max, Q1, Q2 (median), Q3 for all 13 features
  - `_check_class_balance()` - Count samples per wine class (0, 1, 2) and log distribution
  - `_detect_outliers_iqr()` - Use IQR method (Q1 - 1.5*IQR, Q3 + 1.5*IQR) to identify outliers per feature
  - `_plot_distributions()` - Generate 4x4 subplot grid of histograms for all 13 features
  - `_plot_correlation_matrix()` - Create seaborn heatmap of feature correlations
  - `_save_raw_data()` - Save DataFrame as parquet
  - `run_eda()` - Main orchestration function that tracks elapsed time
- **Elapsed Time Tracking**: Record start time at beginning of `run_eda()`, log total execution time at completion using format `"Elapsed time: X.XX seconds"`
- **Outputs**:
  - `output/distributions.png`
  - `output/correlation_matrix.png`
  - `output/wine_raw.parquet`
- **Dependencies**: Step 1

### Step 3: Create `02_feature_engineering.py` - Feature Engineering

- **File**: `part1_claude_code/src/02_feature_engineering.py`
- **Action**: Implement feature engineering with:
  - `_load_raw_data()` - Load from `output/wine_raw.parquet`
  - `_create_derived_features()` - Add 3 new features:
    - `alcohol_to_malic_acid`: alcohol / malic_acid ratio
    - `total_phenols_flavanoids_interaction`: total_phenols * flavanoids
    - `color_intensity_hue_ratio`: color_intensity / hue
  - `_handle_infinite_values()` - Replace inf/nan with column medians
  - `_log_feature_statistics()` - Log mean/std before and after scaling
  - `_scale_features()` - Apply StandardScaler to all numeric features (excluding target)
  - `_stratified_split()` - Use `train_test_split` with `stratify=y`, test_size=0.2, random_state=42
  - `_save_splits()` - Save train.parquet and test.parquet (features + target combined)
  - `run_feature_engineering()` - Main orchestration function
- **Outputs**:
  - `output/train.parquet`
  - `output/test.parquet`
- **Dependencies**: Step 2 (requires `output/wine_raw.parquet`)

### Step 4: Create `03_xgboost_model.py` - Model Training & Evaluation

- **File**: `part1_claude_code/src/03_xgboost_model.py`
- **Action**: Implement XGBoost classifier with hyperparameter tuning:
  - `_load_train_test_data()` - Load train.parquet and test.parquet, split into X and y
  - `_run_hyperparameter_tuning()` - RandomizedSearchCV with:
    - XGBClassifier as base estimator
    - 5-fold StratifiedKFold cross-validation
    - 20 iterations
    - Parameter grid: n_estimators [50,100,200,300], max_depth [3,5,7,9], learning_rate [0.01,0.05,0.1,0.2], subsample [0.7,0.8,0.9,1.0]
    - Scoring: accuracy
  - `_compute_metrics()` - Calculate accuracy, precision (macro), recall (macro), F1-score (macro)
  - `_plot_confusion_matrix()` - Generate heatmap with seaborn
  - `_plot_feature_importance()` - Horizontal bar chart of feature importances
  - `_save_model()` - Pickle best model
  - `_save_tuning_results()` - Save CV results to JSON
  - `_save_evaluation_metrics()` - Save final metrics to JSON
  - `run_model_training()` - Main orchestration function
- **Outputs**:
  - `output/confusion_matrix.png`
  - `output/feature_importance.png`
  - `output/xgb_model.pkl`
  - `output/tuning_results.json`
  - `output/evaluation_metrics.json`
- **Dependencies**: Step 3 (requires `output/train.parquet`, `output/test.parquet`)

### Step 5: Lint and Verify All Scripts

- **Files**: All 3 Python scripts
- **Action**: Run `uv run ruff check --fix` and `uv run python -m py_compile` on each script
- **Dependencies**: Steps 2-4

## Technical Decisions

### Libraries and Tools
- **polars**: For data manipulation (per CLAUDE.md requirements)
- **sklearn.datasets.load_wine()**: Data source
- **sklearn.preprocessing.StandardScaler**: Feature scaling
- **sklearn.model_selection.train_test_split**: Stratified splitting with `stratify` parameter
- **sklearn.model_selection.RandomizedSearchCV**: Hyperparameter tuning with StratifiedKFold
- **sklearn.metrics**: accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
- **xgboost.XGBClassifier**: Classification model
- **matplotlib/seaborn**: Visualization
- **pickle**: Model serialization (using .pkl extension as specified)
- **json**: Results serialization

### Data Structures
- Use polars DataFrames throughout for data manipulation
- Convert to numpy arrays only when interfacing with sklearn/xgboost
- Store train/test data as parquet files with both features and target in same file

### Constants (Top of Each File)
```python
# 01_eda.py
OUTPUT_DIR: str = "output"
FIGURE_DPI: int = 150
IQR_MULTIPLIER: float = 1.5

# 02_feature_engineering.py
OUTPUT_DIR: str = "output"
RAW_DATA_FILE: str = "wine_raw.parquet"
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42
TARGET_COLUMN: str = "target"

# 03_xgboost_model.py
OUTPUT_DIR: str = "output"
MODEL_FILENAME: str = "xgb_model.pkl"
FIGURE_DPI: int = 150
RANDOM_STATE: int = 42
CV_FOLDS: int = 5
N_ITER_SEARCH: int = 20
PARAM_DISTRIBUTIONS: dict = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
}
```

### Code Organization
- Private functions (prefixed with `_`) placed at top of file after constants
- Public `run_*()` function at bottom
- Functions kept under 50 lines
- Two blank lines between function definitions
- Type annotations on all parameters (one per line for multi-param functions)
- Logging using the exact format from CLAUDE.md

### Trade-offs Considered
1. **Parquet vs CSV**: Using parquet for efficient storage and type preservation
2. **pickle vs joblib**: Using pickle (.pkl) as specified in requirements, though joblib would be more robust for large models
3. **Combined vs separate parquet files**: Storing features and target together in train.parquet/test.parquet for simplicity (different from demo which uses separate x_train, y_train files)

## Testing Strategy

1. **Syntax Verification**: Run `uv run python -m py_compile <script>` on each file
2. **Linting**: Run `uv run ruff check --fix <script>` to ensure code style compliance
3. **Sequential Execution**:
   ```bash
   uv run python part1_claude_code/src/01_eda.py
   uv run python part1_claude_code/src/02_feature_engineering.py
   uv run python part1_claude_code/src/03_xgboost_model.py
   ```
4. **Output Verification**: Check that all expected files exist in `output/`:
   - distributions.png, correlation_matrix.png, wine_raw.parquet (from script 1)
   - train.parquet, test.parquet (from script 2)
   - confusion_matrix.png, feature_importance.png, xgb_model.pkl, tuning_results.json, evaluation_metrics.json (from script 3)

## Expected Output

### Directory Structure After Implementation
```
lab03/
├── part1_claude_code/
│   └── src/
│       ├── 01_eda.py
│       ├── 02_feature_engineering.py
│       └── 03_xgboost_model.py
└── output/
    ├── wine_raw.parquet
    ├── distributions.png
    ├── correlation_matrix.png
    ├── train.parquet
    ├── test.parquet
    ├── confusion_matrix.png
    ├── feature_importance.png
    ├── xgb_model.pkl
    ├── tuning_results.json
    └── evaluation_metrics.json
```

### JSON Output Formats

**tuning_results.json**:
```json
{
  "best_params": {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8
  },
  "best_cv_accuracy": 0.9722,
  "n_iterations": 20,
  "cv_folds": 5,
  "all_results": [...]
}
```

**evaluation_metrics.json**:
```json
{
  "accuracy": 0.9722,
  "precision_macro": 0.9735,
  "recall_macro": 0.9712,
  "f1_macro": 0.9720
}
```

### Wine Dataset Details
- **Features (13)**: alcohol, malic_acid, ash, alcalinity_of_ash, magnesium, total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins, color_intensity, hue, od280/od315_of_diluted_wines, proline
- **Target**: 3 classes (0, 1, 2) representing different wine cultivars
- **Samples**: 178 total
