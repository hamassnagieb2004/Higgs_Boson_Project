# Higgs Boson Classification --- End-to-End Machine Learning

A machine learning classification project that distinguishes **Higgs
signal events (`s`)** from **background events (`b`)** using
collision-event features.

The project is implemented in a Jupyter Notebook and covers exploratory
data analysis, data cleaning, leakage-safe preprocessing, model
comparison, hyperparameter optimization with Optuna, ensemble learning,
evaluation, and test predictions.

## Project Structure

Arrange the files as follows:

``` text
Higgs_Boson_Project/
│
├── data/
│   ├── training.csv
│   └── test.csv
│
├── Higgs_Boson_End_to_End_ML_Project.ipynb
├── README.md
├── requirements.txt        # optional: add your pinned dependencies
├── models/                 # generated when models are saved
└── results/                # generated prediction/results files
```

> The notebook expects the CSV files at `data/training.csv` and
> `data/test.csv`, relative to the notebook's working directory.

## Dataset Overview

The dataset contains measurements describing particle-collision events
at the Large Hadron Collider.

-   **Signal (`s`)**: an event classified as a Higgs signal.
-   **Background (`b`)**: an event classified as background.
-   **Target column**: `Label`.
-   **Features**: event-level measurements, including kinematic
    properties of taus, leptons, missing transverse energy, and jets.

The training file must contain the `Label` column. The test file is used
to generate predictions; if it has no labels, supervised evaluation
metrics cannot be calculated for it.

### Important Dataset Notes

-   The value `-999` is treated as a missing-value marker by the
    notebook.
-   `Weight` and `EventId`, if present, are excluded from model input
    features.
-   Missing values are imputed inside the preprocessing Pipeline.
-   Outliers are investigated rather than automatically removed. Extreme
    measurements may represent valid rare physical events.
-   The official test set is not used for model selection.

## Workflow

The notebook is organized into the following stages:

1.  **Setup and data loading** --- import libraries, configure paths,
    and load the CSV files.
2.  **Data understanding and quality checks** --- inspect dimensions,
    data types, target distribution, and missing values.
3.  **Exploratory Data Analysis** --- examine feature distributions and
    investigate outliers and skewness.
4.  **Train/validation split** --- create a stratified holdout split.
5.  **Preprocessing Pipelines** --- handle missing values and apply
    scaling where appropriate, with preprocessing fitted only on
    training folds.
6.  **Baseline model comparison** --- evaluate multiple classification
    families.
7.  **Optuna optimization** --- search for promising hyperparameters
    using cross-validation and ROC-AUC.
8.  **Validation evaluation** --- compare tuned models using metrics
    such as ROC-AUC, PR-AUC, F1-score, recall, precision, and balanced
    accuracy.
9.  **Ensemble learning** --- compare Voting and Stacking classifiers
    with individual models.
10. **Final training and prediction** --- refit the selected pipeline
    and generate predictions for the test dataset.

## Models

The notebook includes models from several classification families,
subject to library availability:

-   **Linear models:** Logistic Regression, SGD Classifier, Ridge
    Classifier.
-   **Distance-based:** K-Nearest Neighbors (KNN).
-   **Support Vector Machines:** SVC.
-   **Tree-based:** Decision Tree, Random Forest, Extra Trees.
-   **Bagging:** Bagging Classifier.
-   **Boosting:** AdaBoost, Gradient Boosting, HistGradientBoosting.
-   **Naive Bayes:** Gaussian Naive Bayes.
-   **Optional boosting libraries:** XGBoost, LightGBM, and CatBoost.
-   **Ensembles:** Voting Classifier and Stacking Classifier.

Not every optional model will be available unless its package is
installed.

## Installation

Create and activate a virtual environment, then install the core
dependencies:

``` bash
python -m venv .venv
```

On Windows:

``` bash
.venv\Scripts\activate
```

Install the required packages:

``` bash
pip install -U numpy pandas matplotlib seaborn scipy scikit-learn optuna joblib jupyter
```

Optional packages for additional boosting models and Optuna
visualizations:

``` bash
pip install -U xgboost lightgbm catboost plotly
```

If you use VS Code, select this virtual environment as the Jupyter
kernel.

## How to Run

1.  Download the dataset files and place them inside the `data/` folder.
2.  Confirm that `training.csv` is not empty and includes the `Label`
    column.
3.  Confirm that `test.csv` is the intended test dataset.
4.  Open `Higgs_Boson_End_to_End_ML_Project.ipynb` in Jupyter Notebook
    or VS Code.
5.  Select the project virtual environment as the notebook kernel.
6.  Run the notebook cells in order.
7.  Review the baseline and Optuna results before choosing the final
    model.

### Runtime Configuration

The notebook defines configurable limits for sampling, cross-validation,
and Optuna trials. Initial settings are intended to make experimentation
more manageable on a local machine.

-   Start with a small number of Optuna trials to verify that the
    workflow runs.
-   Increase the number of trials only after confirming that the models
    and data paths work correctly.
-   SVC and KNN can be computationally expensive on large datasets.
-   Optional boosting libraries can be skipped if they are not
    installed.

## Evaluation

The project reports several complementary classification metrics:

-   **Accuracy** --- overall proportion of correct predictions.
-   **Balanced Accuracy** --- average recall across classes.
-   **Precision** --- how many predicted positive events are actually
    positive.
-   **Recall** --- how many actual positive events are detected.
-   **F1-score** --- harmonic mean of precision and recall.
-   **ROC-AUC** --- ability to rank signal events above background
    events.
-   **PR-AUC** --- precision-recall performance, especially useful when
    class frequencies differ.

The confusion matrix and ROC / Precision-Recall curves provide
additional insight. Model selection should consider the project
objective and more than one metric, rather than accuracy alone.

## Reproducibility and Data Leakage

-   A fixed random seed is used for reproducibility.
-   The train/validation split is created before model fitting.
-   Imputation and scaling are part of each Pipeline, so they are
    learned from the relevant training fold.
-   Optuna searches use training data and cross-validation; the held-out
    validation set is reserved for evaluation.
-   The final selected pipeline is refitted on the available labeled
    training data before test predictions are generated.

## Important Scientific Caution

Do not automatically remove every point flagged as an outlier by the IQR
rule. In a particle-physics dataset, extreme values may be physically
meaningful and may contain useful information for identifying rare
signal events. Investigate the feature definition and event context
before deciding to remove or cap observations.

## Output

The notebook's final section saves the selected model/pipeline and test
predictions in the project output folders. Check the final output paths
printed by the notebook after execution.

## Author

Add your name, contact information, and project links here.
