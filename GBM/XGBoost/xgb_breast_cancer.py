import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
EDA_DIR = os.path.join(RESULTS_DIR, "eda")


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(EDA_DIR, exist_ok=True)

def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y, data.feature_names, data.target_names


def train_test_split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test

def perform_eda(X, y, feature_names, target_names):
    sns.set(style="whitegrid")

    class_counts = y.value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.xticks(ticks=[0, 1], labels=target_names)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "class_distribution.png"))
    plt.close()

    corr = X.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "correlation_heatmap.png"))
    plt.close()

    for col in feature_names:
        plt.figure(figsize=(5, 4))
        sns.histplot(X[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        safe_col_name = col.replace(" ", "_").replace("/", "_")
        plt.savefig(os.path.join(EDA_DIR, f"hist_{safe_col_name}.png"))
        plt.close()


def evaluate_classification(y_true, y_pred, y_proba=None, average="binary"):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average),
        "recall": recall_score(y_true, y_pred, average=average),
        "f1": f1_score(y_true, y_pred, average=average),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
    return cm


def plot_feature_importances(model, feature_names, top_n=15, save_path=None):
    """
    Plot top_n feature importances from an XGBoost model.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(8, 6))
    plt.barh(
        range(len(indices)),
        importances[indices][::-1],
    )
    plt.yticks(
        range(len(indices)),
        [feature_names[i] for i in indices][::-1],
    )
    plt.xlabel("Feature importance")
    plt.title(f"Top {top_n} Feature Importances (XGBoost)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def save_classification_report(report_str, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_str)

def main():
    ensure_results_dir()

    X, y, feature_names, target_names = load_data()
    perform_eda(X, y, feature_names, target_names)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    baseline_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=2000)),
        ]
    )
    baseline_model.fit(X_train, y_train)

    y_pred_baseline = baseline_model.predict(X_test)
    y_proba_baseline = baseline_model.predict_proba(X_test)[:, 1]

    baseline_metrics = evaluate_classification(
        y_true=y_test,
        y_pred=y_pred_baseline,
        y_proba=y_proba_baseline,
        average="binary",
    )

    cm_baseline = plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred_baseline,
        class_names=target_names,
        save_path=os.path.join(PLOTS_DIR, "confusion_matrix_baseline.png"),
    )
    np.savetxt(
        os.path.join(RESULTS_DIR, "confusion_matrix_baseline.csv"),
        cm_baseline,
        delimiter=",",
        fmt="%d",
    )

    baseline_report = classification_report(
        y_test, y_pred_baseline, target_names=target_names
    )
    save_classification_report(
        baseline_report,
        os.path.join(REPORTS_DIR, "classification_report_baseline.txt"),
    )

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    xgb_model.fit(X_train, y_train)

    y_pred_xgb = xgb_model.predict(X_test)
    y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

    xgb_metrics = evaluate_classification(
        y_true=y_test,
        y_pred=y_pred_xgb,
        y_proba=y_proba_xgb,
        average="binary",
    )

    cm_xgb = plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred_xgb,
        class_names=target_names,
        save_path=os.path.join(PLOTS_DIR, "confusion_matrix_xgboost.png"),
    )
    np.savetxt(
        os.path.join(RESULTS_DIR, "confusion_matrix_xgboost.csv"),
        cm_xgb,
        delimiter=",",
        fmt="%d",
    )

    xgb_report = classification_report(
        y_test, y_pred_xgb, target_names=target_names
    )
    save_classification_report(
        xgb_report,
        os.path.join(REPORTS_DIR, "classification_report_xgboost.txt"),
    )

    plot_feature_importances(
        model=xgb_model,
        feature_names=feature_names,
        top_n=15,
        save_path=os.path.join(PLOTS_DIR, "feature_importances_xgboost.png"),
    )

    print("=== Baseline: Logistic Regression (Standardized) ===")
    print(baseline_metrics)
    print()
    print("=== XGBoost Model ===")
    print(xgb_metrics)
    print()

    print("=== Classification Report (Baseline) ===")
    print(baseline_report)
    print()
    print("=== Classification Report (XGBoost) ===")
    print(xgb_report)

    metrics_df = pd.DataFrame(
        [baseline_metrics, xgb_metrics],
        index=["baseline_logreg", "xgboost"],
    )
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "metrics_comparison.csv"), index=True)

    print("\nAll artifacts saved under the 'results' directory next to this script:")
    print(f"- EDA plots       → {EDA_DIR}")
    print(f"- Confusion plots → {PLOTS_DIR}")
    print(f"- Confusion CSVs  → {RESULTS_DIR}")
    print(f"- Metrics table   → {os.path.join(RESULTS_DIR, 'metrics_comparison.csv')}")
    print(f"- Reports         → {REPORTS_DIR}")


if __name__ == "__main__":
    main()
