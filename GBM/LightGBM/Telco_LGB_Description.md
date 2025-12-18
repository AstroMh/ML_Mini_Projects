# LightGBM Customer Churn Mini-Project  
Machine Learning Mini-Projects Repository

## Overview
This mini-project explores **customer churn prediction** using **LightGBM**, a high-performance gradient boosting framework optimized for speed and scalability on tabular data.

The project was intentionally designed not only to achieve good performance, but also to **demonstrate the practical behavior of LightGBM**, including its sensitivity to hyperparameters, evaluation setup, and overfitting.

A simple **Logistic Regression** model is used as a baseline, and the project documents how:
- an improperly configured LightGBM model initially underperformed the baseline,
- early stopping and regularization were necessary to control overfitting,
- careful tuning led to significant improvements in recall and F1-score,
- trade-offs between recall, precision, and accuracy naturally emerged.

This mirrors real-world machine learning workflows rather than idealized benchmark results.

---

## Dataset Description

### Dataset Name
Telco Customer Churn Dataset

### Source
Originally published by IBM and distributed via Kaggle.  
The dataset represents a real-world telecommunications churn problem.

### Task
Binary classification

### Objective
Predict whether a customer will churn (cancel service) based on demographic, service usage, and billing information.

### Target Variable
Churn:
- 0 → No
- 1 → Yes

### Dataset Size
- 7,043 customers
- 21 columns (20 features + 1 target)

After minimal cleaning:
- No missing values
- All features usable for modeling

---

### Feature Overview

The dataset contains a **mix of categorical and numerical features**, making it ideal for LightGBM.

#### Numerical Features
- tenure — number of months the customer has stayed
- MonthlyCharges — monthly billing amount
- TotalCharges — total billing amount over time

#### Categorical Features
- gender
- Partner
- Dependents
- PhoneService
- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies
- Contract
- PaperlessBilling
- PaymentMethod

This structure closely reflects real business data and highlights the importance of correct categorical handling.

---

## Project Structure

LightGBM/  
• lightgbm_main.py  
• requirements.txt  
• data/  
 • telco_churn.csv  
• results/  
 • plots/  
  - churn_distribution.png  
  - tenure_vs_churn.png  
  - monthly_charges_vs_churn.png  
  - contract_vs_churn.png  
  - confusion_matrix_baseline_logreg.png  
  - confusion_matrix_lightgbm.png  
  - feature_importance_lightgbm.png  
 • reports/  
  - classification_report_baseline_logreg.txt  
  - classification_report_lightgbm.txt  
 • metrics_baseline_logreg.csv  
 • metrics_lightgbm.csv  
 • metrics_comparison.csv  

All artifacts are generated automatically when the script is executed.

---

## Exploratory Data Analysis (EDA)

EDA was intentionally kept lightweight and business-focused:

- Churn distribution confirmed class imbalance.
- Tenure showed strong separation between churners and non-churners.
- MonthlyCharges revealed non-linear relationships with churn.
- Contract type demonstrated one of the strongest categorical signals.

These observations motivated the use of gradient boosting rather than linear models.

---

## Baseline Model: Logistic Regression

A Logistic Regression model was chosen as the baseline because:

- It is a common first model in churn prediction.
- It provides a clear and interpretable reference point.
- It highlights whether a more complex model is truly needed.

Categorical features were one-hot encoded, and numerical features were standardized.

### Baseline Results
- Moderate accuracy
- Weak recall and F1-score for churners
- Difficulty capturing non-linear patterns

This behavior is expected for linear models on churn data.

---

## LightGBM Model

### Initial Attempt and Overfitting
The first LightGBM model was trained **without early stopping** and with minimal regularization.

Result:
- Training performance improved continuously.
- Test-set performance degraded.
- The model performed **worse than the Logistic Regression baseline**.

This highlighted an important lesson:

**LightGBM is powerful but sensitive.  
Without proper control, it can easily overfit on small to medium datasets.**

---

### Model Refinement
To address overfitting, the following were added:

- Early stopping using validation loss
- Reduced tree complexity
- Minimum samples per leaf
- Feature and row subsampling
- Class imbalance handling using scale_pos_weight

After these changes:
- Recall and F1-score improved significantly
- Precision and accuracy decreased slightly
- Overall churn detection improved meaningfully

This trade-off is expected and often desirable in churn problems, where missing churners is costly.

---

## Evaluation Metrics

Both models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Full Classification Report

Metrics were saved to disk for transparency and reproducibility.

---

## Key Findings

- Logistic Regression provides a strong but limited baseline.
- LightGBM is highly sensitive to hyperparameters and training setup.
- Without early stopping, LightGBM can overfit and underperform.
- With proper tuning, LightGBM significantly improves recall and F1-score.
- Accuracy alone is not a sufficient metric for churn prediction.
- Threshold choice and metric alignment strongly affect perceived performance.

This project demonstrates why **model configuration and evaluation strategy matter as much as model choice**.

---

## What This Project Demonstrates

- Realistic ML experimentation, including failure and correction
- Proper use of baselines
- Overfitting detection and mitigation
- Practical handling of categorical features
- Business-oriented evaluation of churn models
- Clean project structure with saved artifacts

---

## Conclusion

This mini-project shows that LightGBM is a powerful but sensitive model.  
Initial results can be misleading if early stopping and regularization are ignored.

With proper tuning and evaluation, LightGBM clearly outperforms linear baselines for customer churn prediction.  
Further improvements are likely achievable with additional hyperparameter tuning and threshold optimization.

---

## Next Steps

- Threshold optimization based on business objectives
- Cross-validation experiments
- Comparison with XGBoost and CatBoost on the same dataset
- Full GBM comparison project across multiple datasets

---
