import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

def get_project_paths(results_dir_name: str = "results"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, results_dir_name)
    os.makedirs(results_dir, exist_ok=True)
    return base_dir, results_dir

def load_data(as_frame: bool = True):
    iris = load_iris(as_frame=as_frame)
    df = iris.frame  # includes feature columns + 'target'
    X = df[iris.feature_names]
    y = df["target"]
    target_names = iris.target_names
    feature_names = iris.feature_names
    return X, y, feature_names, target_names, df


def explore_data(df: pd.DataFrame, results_dir: str) -> None:
    """
    Basic exploratory visualization:
      - pairplot
      - correlation heatmap
    """
    sns.set(style="whitegrid")

    # Use a copy with a nicer target name
    df_plot = df.copy()
    df_plot["species"] = df_plot["target"].map({
        0: "setosa",
        1: "versicolor",
        2: "virginica",
    })

    feature_cols = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]

    # Pairplot
    pairplot_path = os.path.join(results_dir, "iris_pairplot.png")
    sns.pairplot(df_plot, vars=feature_cols, hue="species", diag_kind="hist")
    plt.tight_layout()
    plt.savefig(pairplot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Pairplot saved to {pairplot_path}")

    # Correlation heatmap
    corr = df[feature_cols].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Iris feature correlation heatmap")
    plt.tight_layout()
    heatmap_path = os.path.join(results_dir, "iris_corr_heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Correlation heatmap saved to {heatmap_path}")


def plot_decision_regions_2d(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model: SVC,
    feature_names: list[str],
    results_dir: str,
    filename: str = "iris_svc_decision_regions.png",
):
    """
    Plot decision regions using two features (petal length & petal width).
    This is purely for visualization; it uses only 2D space.
    """
    try:
        idx_pl = feature_names.index("petal length (cm)")
        idx_pw = feature_names.index("petal width (cm)")
    except ValueError:
        print("[WARN] petal length/width not found in feature_names; skipping decision region plot.")
        return

    X_train_2d = X_train[:, [idx_pl, idx_pw]]
    X_test_2d = X_test[:, [idx_pl, idx_pw]]

    # Refitting a small SVC on just these 2 features (for visualization only)
    clf_vis = SVC(kernel="rbf", gamma="scale", C=1.0)
    clf_vis.fit(X_train_2d, y_train)

    x_min, x_max = X_train_2d[:, 0].min() - 0.5, X_train_2d[:, 0].max() + 0.5
    y_min, y_max = X_train_2d[:, 1].min() - 0.5, X_train_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf_vis.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")

    scatter_train = plt.scatter(
        X_train_2d[:, 0],
        X_train_2d[:, 1],
        c=y_train,
        cmap="viridis",
        edgecolor="k",
        marker="o",
        label="Train",
        alpha=0.8,
    )
    scatter_test = plt.scatter(
        X_test_2d[:, 0],
        X_test_2d[:, 1],
        c=y_test,
        cmap="viridis",
        edgecolor="k",
        marker="^",
        label="Test",
        alpha=0.9,
    )

    plt.xlabel("petal length (cm)")
    plt.ylabel("petal width (cm)")
    plt.title("SVC decision regions (2D: petal length vs petal width)")
    plt.legend(handles=[scatter_train, scatter_test])
    plt.tight_layout()

    path = os.path.join(results_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Decision region plot saved to {path}")

def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
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
    Standardizing features using StandardScaler.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def train_svc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str | float = "scale",
):

    svc = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        probability=False,
        random_state=42,
    )
    svc.fit(X_train, y_train)
    return svc


def evaluate_classifier(
    model,
    X_test,
    y_test,
    target_names,
    results_dir: str,
    report_filename: str = "iris_svc_report.txt",
    cm_filename: str = "iris_svc_confusion_matrix.png",
):

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=target_names)

    print("\n=== Iris – SVC Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(cr)

    # Save text report
    report_path = os.path.join(results_dir, report_filename)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Iris – SVC Evaluation\n\n")
        f.write(f"Accuracy: {acc:.6f}\n\n")
        f.write("Confusion matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification report:\n")
        f.write(cr)

    print(f"[INFO] Evaluation report saved to {report_path}")

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix – Iris SVC")
    plt.tight_layout()
    cm_path = os.path.join(results_dir, cm_filename)
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix heatmap saved to {cm_path}")

    return y_pred, {"accuracy": acc, "confusion_matrix": cm, "classification_report": cr}


def main():
    _, results_dir = get_project_paths(results_dir_name="results")

    X, y, feature_names, target_names, df_full = load_data(as_frame=True)
    explore_data(df_full, results_dir=results_dir)
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=0.2, random_state=42
    )

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    svc_model = train_svc(
        X_train_scaled,
        y_train,
        kernel="rbf",
        C=1.0,
        gamma="scale",
    )

    y_pred, metrics = evaluate_classifier(
        svc_model,
        X_test_scaled,
        y_test,
        target_names=target_names,
        results_dir=results_dir,
        report_filename="iris_svc_report.txt",
        cm_filename="iris_svc_confusion_matrix.png",
    )

    plot_decision_regions_2d(
        X_train_scaled,
        y_train.to_numpy(),
        X_test_scaled,
        y_test.to_numpy(),
        svc_model,
        feature_names=feature_names,
        results_dir=results_dir,
        filename="iris_svc_decision_regions.png",
    )


if __name__ == "__main__":
    main()
