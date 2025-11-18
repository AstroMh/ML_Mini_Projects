import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

def get_project_paths(data_filename: str = "advertising.csv",
                      results_dir_name: str = "results"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, data_filename)
    results_dir = os.path.join(base_dir, results_dir_name)
    os.makedirs(results_dir, exist_ok=True)
    return data_path, results_dir


def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    return df

def explore_data(df: pd.DataFrame, results_dir: str) -> None:
    # Keep only numeric columns for pairplot
    numeric_cols = [
        "Daily Time Spent on Site",
        "Age",
        "Area Income",
        "Daily Internet Usage",
        "Male",
        "Clicked on Ad",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    subset = df[numeric_cols]

    pairplot_path = os.path.join(results_dir, "advertising_pairplot.png")
    sns.pairplot(subset, hue="Clicked on Ad")
    plt.tight_layout()
    plt.savefig(pairplot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Pairplot saved to {pairplot_path}")

    corr = subset.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Advertising – Correlation Heatmap")
    plt.tight_layout()
    heatmap_path = os.path.join(results_dir, "advertising_corr_heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Correlation heatmap saved to {heatmap_path}")


def prepare_features(df: pd.DataFrame):
    feature_cols = [
        "Daily Time Spent on Site",
        "Age",
        "Area Income",
        "Daily Internet Usage",
        "Male",
    ]
    target_col = "Clicked on Ad"

    # Filter out rows with missing values in the selected columns
    cols_needed = feature_cols + [target_col]
    df_clean = df[cols_needed].dropna()

    X = df_clean[feature_cols]
    y = df_clean[target_col].astype(int)

    return X, y, feature_cols


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


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Standardize features using StandardScaler (fit on train, transform both).
    """
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_classification(
    model,
    X_test,
    y_test,
    results_dir: str,
    report_filename: str = "advertising_logistic_report.txt",
    cm_filename: str = "advertising_confusion_matrix.png",
):
    """
    Evaluate a classification model and save metrics + confusion matrix.

    Saves:
      - Text report (accuracy + classification report + confusion matrix)
      - Confusion matrix heatmap as PNG
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    print("\n=== Advertising – Logistic Regression Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(cr)

    report_path = os.path.join(results_dir, report_filename)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Advertising – Logistic Regression Evaluation\n\n")
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
    plt.title("Confusion Matrix – Advertising Logistic Regression")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix heatmap saved to {cm_path}")

    return y_pred, {"accuracy": acc, "confusion_matrix": cm, "classification_report": cr}


def plot_roc_curve(
    model,
    X_test,
    y_test,
    results_dir: str,
    filename: str = "advertising_roc_curve.png",
):
    # Predict probabilities for positive class (1 = clicked)
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Advertising Logistic Regression")
    plt.legend(loc="lower right")
    plt.tight_layout()

    roc_path = os.path.join(results_dir, filename)
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] ROC curve saved to {roc_path}")

    return fpr, tpr, roc_auc


def save_coefficients_table(
    model,
    feature_names,
    results_dir: str,
    filename_csv: str = "advertising_logistic_coefficients.csv",
    filename_txt: str = "advertising_logistic_coefficients.txt",
):
    coeffs = model.coef_.ravel()
    coeff_df = pd.DataFrame(
        {"feature": feature_names, "coefficient": coeffs}
    ).sort_values(by="coefficient", ascending=False)

    csv_path = os.path.join(results_dir, filename_csv)
    coeff_df.to_csv(csv_path, index=False)

    txt_path = os.path.join(results_dir, filename_txt)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Advertising – Logistic Regression Coefficients\n\n")
        for _, row in coeff_df.iterrows():
            f.write(f"{row['feature']}: {row['coefficient']:.6f}\n")

    print(f"[INFO] Coefficients saved to {csv_path} and {txt_path}")

def main():
    # Resolve paths
    data_path, results_dir = get_project_paths(
        data_filename="advertising.csv",
        results_dir_name="results",
    )

    df = load_data(data_path)

    explore_data(df, results_dir=results_dir)

    X, y, feature_names = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=0.3, random_state=42
    )
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    model = train_logistic_regression(X_train_scaled, y_train)

    y_pred, metrics = evaluate_classification(
        model,
        X_test_scaled,
        y_test,
        results_dir=results_dir,
        report_filename="advertising_logistic_report.txt",
        cm_filename="advertising_confusion_matrix.png",
    )
    fpr, tpr, roc_auc = plot_roc_curve(
        model,
        X_test_scaled,
        y_test,
        results_dir=results_dir,
        filename="advertising_roc_curve.png",
    )
    save_coefficients_table(model, feature_names, results_dir=results_dir)


if __name__ == "__main__":
    main()
