# Machine Learning Practice Projects (Python)

This repository collects my personal **machine learning** and **computer vision** practice projects implemented in Python.

The goal of this repository is straightforward:
work with real datasets, build clear baseline models, experiment with more advanced methods when appropriate, and keep everything organized in **clean, reproducible `.py` scripts** rather than scattered or unfinished notebooks.

Each project focuses on understanding *why* a model behaves the way it does, not just on achieving a high score.

---

## 🔍 What’s inside

- **Self-contained mini-projects**  
  Each project lives in its own folder and typically includes:
  - A main Python script (often converted from an exploratory notebook)
  - A local `results/` directory with saved plots and evaluation metrics
  - A dataset (CSV or sklearn-provided), or clear instructions for downloading it
  - A project-specific `README.md` describing the task, approach, and findings

- **Gradient Boosting Machines (GBM) series**  
  A dedicated `GBM/` folder contains focused mini-projects comparing modern boosting frameworks:
  - XGBoost
  - CatBoost
  - LightGBM  

  Each GBM project follows the same structure and emphasizes:
  - baseline comparison,
  - model behavior and sensitivity,
  - proper evaluation and overfitting control.

- **Topics covered (so far)**  
  The repository is actively growing and currently includes projects on:
  - **k-Nearest Neighbors (KNN)** – tabular data and handwritten digits
  - **Linear regression** – house prices, e-commerce spending
  - **Logistic regression** – classification tasks such as ad clicks and churn
  - **Random forests** – tabular classification and regression
  - **Gradient boosting** – XGBoost, CatBoost, LightGBM
  - **Model evaluation** – train/test splits, accuracy, recall, F1-score, ROC-AUC, confusion matrices
  - **Data preprocessing & feature engineering** – scaling, encoding, feature selection
  - **Data visualization** – distributions, correlations, and model diagnostics

More advanced topics (cross-validation, hyperparameter tuning, interpretability, and computer vision tasks) are added incrementally.

---

## 🗂 Project structure

Most projects follow a consistent structure:

- `<project_name>/`
  - `main_script.py` – complete training and evaluation pipeline
  - `results/` – saved plots, metrics, and model outputs
  - `README.md` – explanation of the problem, methodology, and results
  - optional data files or download instructions

Each project can be run independently to reproduce its results.

---

## 📚 Why this repo exists

This repository serves as a **learning log and technical portfolio**:

- To practice core machine learning concepts on realistic datasets
- To develop disciplined, reproducible ML workflows
- To document not only successful results, but also model limitations, failures, and corrections
- To build a growing collection of well-structured reference implementations for common ML problems

The emphasis is on clarity, correctness, and understanding — not leaderboard chasing.
