# CatBoost Classification Mini-Project  

## Overview
This mini-project demonstrates the use of **CatBoost**, a gradient boosting algorithm specifically designed to handle categorical features efficiently, on a real-world structured dataset.  

The main goal of the project is to show:
- how CatBoost works with raw categorical features without manual encoding,
- how it compares to a classical baseline model,
- and why CatBoost is especially powerful for tabular datasets with mixed feature types.

The project follows a clean end-to-end workflow:
- dataset loading and minimal cleaning,
- basic exploratory data analysis (EDA),
- baseline model training (Decision Tree),
- CatBoost model training,
- quantitative and visual comparison of results,
- automatic saving of all artifacts (plots, reports, metrics).

---

## Dataset Description

### Dataset Name
Adult Income Dataset (also known as Census Income Dataset)

### Source
Loaded via scikit-learn using the OpenML interface:
- fetch_openml(name="adult", version=2)

Original dataset collected from the 1994 U.S. Census Bureau database.

### Task
Binary classification

### Objective
Predict whether a person’s annual income exceeds \$50,000 based on demographic and employment-related features.

### Target Variable
income (binary):
- <=50K
- >50K

### Features
The dataset contains a mix of **numerical and categorical features**, including:

Numerical features (examples):
- age
- education-num
- hours-per-week
- capital-gain
- capital-loss

Categorical features (examples):
- workclass
- education
- marital-status
- occupation
- relationship
- race
- sex
- native-country

This mix makes the dataset ideal for demonstrating CatBoost’s native categorical handling.

### Dataset Size
After minimal cleaning (removal of missing values):
- Approximately 45,000 samples
- 14 input features
- Binary target

---

## Project Structure

CatBoost/  
• catboost_adult.ipynb or catboost_main.py  
• requirements.txt  
• results/  
 • plots/  
  - target_distribution.png  
  - hist_age.png (and other numeric histograms)  
  - cat_education.png (and other categorical plots)  
  - confusion_matrix_baseline.png  
  - confusion_matrix_catboost.png  
  - feature_importances_catboost.png  
 • reports/  
  - classification_report_baseline.txt  
  - classification_report_catboost.txt  
 • confusion_matrix_baseline.csv  
 • confusion_matrix_catboost.csv  
 • metrics_baseline.csv  
 • metrics_catboost.csv  
 • metrics_comparison.csv  
 • feature_importances_top15.csv  

All files are generated automatically when the script or notebook is executed.

---

## Exploratory Data Analysis (EDA)

The EDA phase is intentionally lightweight and focuses on understanding the data quickly:

- Target distribution to inspect class imbalance.
- Histograms for selected numerical features to observe scale and skewness.
- Category frequency plots for selected categorical features.
- Missing-value summary saved as a CSV file.

This level of EDA is sufficient for a mini-project while keeping the notebook clean and focused on modeling.

---

## Models

### Baseline Model: Decision Tree
A Decision Tree classifier is used as a baseline model.

Reasons for choosing it:
- Simple and interpretable.
- Can model non-linear relationships.
- Serves as a strong classical reference point.

Because Decision Trees cannot handle string categories directly, categorical features are encoded using **One-Hot Encoding** inside a preprocessing pipeline.

The baseline model establishes a reference performance level.

---

### Main Model: CatBoost Classifier

CatBoost is the primary focus of this project.

Key characteristics:
- Handles categorical features natively.
- Uses ordered target statistics to avoid target leakage.
- Requires minimal preprocessing.
- Performs especially well on tabular datasets.

The model is trained using:
- Raw categorical columns (no manual encoding),
- Logloss as the optimization objective,
- AUC as an evaluation metric,
- Early stopping on a validation set.

This setup highlights CatBoost’s advantage over classical approaches.

---

## Evaluation Metrics

Both models are evaluated using the same metrics for a fair comparison:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC (for CatBoost)
- Confusion Matrix
- Full Classification Report

All metrics are saved to disk, and a comparison table is created.

---

## Results Summary

The comparison clearly shows that:

- CatBoost outperforms the baseline Decision Tree across most evaluation metrics.
- The improvement is especially visible in recall and ROC-AUC.
- Native categorical handling allows CatBoost to capture complex feature interactions without extensive preprocessing.
- Feature importance analysis highlights which demographic and employment features contribute most to income prediction.

This confirms CatBoost’s strength on structured datasets with categorical variables.

---

## What This Project Demonstrates

- A clean and minimal ML workflow suitable for real-world tabular data.
- Proper use of baseline models for comparison.
- Practical advantages of CatBoost over classical tree models.
- Automatic generation and organization of ML artifacts.
- Reproducible experiments with fixed random seeds.

---

## Next Steps

This mini-project is part of a broader series focused on gradient boosting methods.

Planned next steps:
- LightGBM mini-project
- Unified large-scale GBM comparison project (XGBoost vs LightGBM vs CatBoost) on a real-world dataset
- Hyperparameter tuning and cross-validation experiments
- Explainability analysis using SHAP values

---

## Author
This project was developed as part of a personal Machine Learning mini-projects repository to deepen practical understanding of modern gradient boosting algorithms on structured data.
