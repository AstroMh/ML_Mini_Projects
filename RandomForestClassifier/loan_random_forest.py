import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # use a non-interactive backend (no GUI needed)

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

def get_project_paths(
    data_filename: str = "loan_data.csv",
    results_dir_name: str = "results",
):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, data_filename)
    results_dir = os.path.join(base_dir, results_dir_name)
    os.makedirs(results_dir, exist_ok=True)
    return data_path, results_dir


def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    return df

def explore_data(df: pd.DataFrame, results_dir: str) -> None:
    """
    Basic exploratory visualization.

    Saves:
      - pairplot of selected numeric features + target
      - correlation heatmap of numeric columns

    Wrapped in try/except so plotting issues (e.g. seaborn/pandas version)
    don't crash the whole script.
    """
    cols_for_pairplot = [
        "fico",
        "int.rate",
        "installment",
        "log.annual.inc",
        "dti",
        "credit.policy",
        "not.fully.paid",
    ]
    available_cols = [c for c in cols_for_pairplot if c in df.columns]

    # ---- Pairplot ----
    try:
        if "not.fully.paid" in available_cols:
            subset = df[available_cols].dropna()
            pairplot_path = os.path.join(results_dir, "loan_pairplot.png")
            sns.pairplot(subset, hue="not.fully.paid")
            plt.tight_layout()
            plt.savefig(pairplot_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[INFO] Pairplot saved to {pairplot_path}")
        else:
            print("[WARN] 'not.fully.paid' not in columns for pairplot; skipping.")
    except Exception as e:
        print(f"[WARN] Failed to create pairplot: {e}")

    # ---- Correlation heatmap ----
    try:
        numeric_df = df.select_dtypes(include=["number"]).dropna()
        if numeric_df.shape[1] > 1:
            corr = numeric_df.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, cmap="coolwarm", center=0, annot=True)
            plt.title("LendingClub – Correlation Heatmap")
            plt.tight_layout()
            heatmap_path = os.path.join(results_dir, "loan_corr_heatmap.png")
            plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[INFO] Correlation heatmap saved to {heatmap_path}")
        else:
            print("[WARN] Not enough numeric columns for heatmap; skipping.")
    except Exception as e:
        print(f"[WARN] Failed to create correlation heatmap: {e}")


def preprocess_data(df: pd.DataFrame):
    """
    Preprocess the LendingClub dataset.

    - One-hot encode the 'purpose' categorical column (if present).
    - Keep all other columns except the target.
    - Define target as 'not.fully.paid' (1 = did NOT pay back in full, 0 = paid).

    Returns
    -------
    X : DataFrame
        Feature matrix after encoding.
    y : Series
        Target vector.
    feature_names : list
        Names of the final feature columns.
    """
    if "not.fully.paid" not in df.columns:
        raise ValueError("Expected target column 'not.fully.paid' not found in DataFrame.")

    if "purpose" in df.columns:
        df = pd.get_dummies(df, columns=["purpose"], drop_first=True)

    df = df.dropna(subset=["not.fully.paid"])

    X = df.drop("not.fully.paid", axis=1)
    y = df["not.fully.paid"].astype(int)

    X = X.dropna()
    y = y.loc[X.index]

    feature_names = X.columns.tolist()

    if X.shape[0] == 0 or X.shape[1] == 0:
        raise RuntimeError("After preprocessing, X is empty. Check your data and preprocessing steps.")

    return X, y, feature_names


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.3,
    random_state: int = 42,
):

    return train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
        stratify=y,
    )


def train_random_forest(
    X_train,
    y_train,
    n_estimators: int = 600,
    max_depth=None,
    random_state: int = 42,
):

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def evaluate_classifier(
    model,
    X_test,
    y_test,
    results_dir: str,
    report_filename: str = "loan_random_forest_report.txt",
    cm_filename: str = "loan_random_forest_confusion_matrix.png",
):

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    print("\n=== LendingClub – Random Forest Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(cr)

    report_path = os.path.join(results_dir, report_filename)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("LendingClub – Random Forest Evaluation\n\n")
        f.write(f"Accuracy: {acc:.6f}\n\n")
        f.write("Confusion matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification report:\n")
        f.write(cr)

    print(f"[INFO] Evaluation report saved to {report_path}")

    cm_path = os.path.join(results_dir, cm_filename)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix – LendingClub Random Forest")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix heatmap saved to {cm_path}")

    return y_pred, {
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": cr,
    }


def plot_roc_curve(
    model,
    X_test,
    y_test,
    results_dir: str,
    filename: str = "loan_random_forest_roc_curve.png",
):

    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Model does not support predict_proba; cannot compute ROC curve.")

    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – LendingClub Random Forest")
    plt.legend(loc="lower right")
    plt.tight_layout()

    roc_path = os.path.join(results_dir, filename)
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] ROC curve saved to {roc_path}")

    return fpr, tpr, roc_auc


def plot_feature_importances(
    model,
    feature_names,
    results_dir: str,
    filename: str = "loan_random_forest_feature_importances.png",
):

    importances = model.feature_importances_
    series = pd.Series(importances, index=feature_names).sort_values()

    if len(series) == 0:
        print("[WARN] No feature importances to plot.")
        return

    height = max(4, len(series) * 0.25)
    plt.figure(figsize=(8, height))
    series.plot(kind="barh")
    plt.xlabel("Feature importance")
    plt.title("Random Forest Feature Importances – LendingClub")
    plt.tight_layout()

    fi_path = os.path.join(results_dir, filename)
    plt.savefig(fi_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Feature importances plot saved to {fi_path}")


def save_feature_importances_table(
    model,
    feature_names,
    results_dir: str,
    filename_csv: str = "loan_random_forest_feature_importances.csv",
    filename_txt: str = "loan_random_forest_feature_importances.txt",
):

    importances = model.feature_importances_
    fi_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values(by="importance", ascending=False)

    csv_path = os.path.join(results_dir, filename_csv)
    fi_df.to_csv(csv_path, index=False)

    txt_path = os.path.join(results_dir, filename_txt)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("LendingClub – Random Forest Feature Importances\n\n")
        for _, row in fi_df.iterrows():
            f.write(f"{row['feature']}: {row['importance']:.6f}\n")

    print(f"[INFO] Feature importances saved to {csv_path} and {txt_path}")

def main():
    # Resolve paths
    data_path, results_dir = get_project_paths(
        data_filename="loan_data.csv",
        results_dir_name="results",
    )

    df = load_data(data_path)

    explore_data(df, results_dir=results_dir)
    X, y, feature_names = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=0.3, random_state=42
    )

    rf_model = train_random_forest(
        X_train,
        y_train,
        n_estimators=600,
        max_depth=None,
    )

    y_pred, metrics = evaluate_classifier(
        rf_model,
        X_test,
        y_test,
        results_dir=results_dir,
        report_filename="loan_random_forest_report.txt",
        cm_filename="loan_random_forest_confusion_matrix.png",
    )

    plot_roc_curve(
        rf_model,
        X_test,
        y_test,
        results_dir=results_dir,
        filename="loan_random_forest_roc_curve.png",
    )

    plot_feature_importances(
        rf_model,
        feature_names,
        results_dir=results_dir,
        filename="loan_random_forest_feature_importances.png",
    )
    save_feature_importances_table(
        rf_model,
        feature_names,
        results_dir=results_dir,
        filename_csv="loan_random_forest_feature_importances.csv",
        filename_txt="loan_random_forest_feature_importances.txt",
    )


if __name__ == "__main__":
    main()
