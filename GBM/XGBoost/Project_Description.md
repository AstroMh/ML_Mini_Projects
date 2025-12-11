# XGBoost Classification Mini-Project  
Machine Learning Mini-Projects Repository

## Overview  
This mini-project demonstrates a complete machine learning workflow using XGBoost for binary classification.  
The workflow includes:

- Exploratory Data Analysis (EDA)  
- Baseline model with Logistic Regression  
- XGBoost model training and evaluation  
- Standard ML performance metrics  
- Confusion matrices and feature importance plots  
- Automatic saving of all generated artifacts into a structured results directory

The dataset used is the Breast Cancer Wisconsin Diagnostic dataset from scikit-learn, a widely used benchmark for structured data models.

---

## Project Structure  

XGBoost/  
• xgboost_main.py  
• requirements.txt  
• results/  
 • eda/  
  - class_distribution.png  
  - correlation_heatmap.png  
  - hist_feature_*.png  
 • plots/  
  - confusion_matrix_baseline.png  
  - confusion_matrix_xgboost.png  
  - feature_importances_xgboost.png  
 • reports/  
  - classification_report_baseline.txt  
  - classification_report_xgboost.txt  
 • confusion_matrix_baseline.csv  
 • confusion_matrix_xgboost.csv  
 • metrics_comparison.csv  

All directories and files are generated automatically when the script is executed.

---

## Dataset  
Name: Breast Cancer Wisconsin Diagnostic  
Source: sklearn.datasets.load_breast_cancer  
Task: Binary classification  
Classes:  
0 = Malignant  
1 = Benign  

The dataset contains 30 numerical features extracted from digitized images of breast tissue masses.

---

## Models Included  

### Baseline Model: Logistic Regression  
Serves as a simple, interpretable starting point to validate the pipeline and produce a performance benchmark.  
Uses StandardScaler and max_iter = 2000 to ensure convergence.

### XGBoost Model: Gradient Boosted Trees  
Key hyperparameters used:  
n_estimators = 300  
max_depth = 4  
learning_rate = 0.05  
subsample = 0.8  
colsample_bytree = 0.8  
tree_method = hist  
objective = binary:logistic  

---

## Evaluation Metrics  

Both models are evaluated using:  

• Accuracy  
• Precision  
• Recall  
• F1-score  
• ROC-AUC  
• Confusion matrix  
• Classification report  

A summary table is exported as:  
results/metrics_comparison.csv

---

## Key Findings  

• XGBoost significantly outperforms Logistic Regression across all key metrics.  
• Feature importance plot highlights the strongest predictive features.  
• Confusion matrices show improved classification performance for both malignant and benign cases.  
• EDA reveals the underlying structure and distribution of the dataset.

---

## What I Learned  

• Designing a full ML pipeline from EDA to model deployment  
• Comparing baseline models against advanced ensemble methods  
• Practical understanding of how XGBoost captures non-linear relationships  
• Saving and organizing ML artifacts for reproducibility  
• Structuring ML projects professionally

---

## Next Steps  

This project is part of a series exploring gradient boosting methods.  
Upcoming mini-projects will cover:

• LightGBM classification  
• CatBoost classification  

After these, a full real-world GBM comparison project will be created using a large external dataset.

