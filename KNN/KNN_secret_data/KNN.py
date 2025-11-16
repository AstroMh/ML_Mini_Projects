import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report


def get_project_paths(results_dir_name: str = "results"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "KNN_Project_Data")
    results_dir = os.path.join(base_dir, results_dir_name)
    os.makedirs(results_dir, exist_ok=True)
    return data_path, results_dir


def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path, index_col=0)
    return df


def explore_data(df: pd.DataFrame, results_dir: str) -> None:
    """
    Basic exploratory visualization: pairplot colored by TARGET CLASS.
    Saves the plot as an image in results_dir.
    """
    sns.pairplot(df, hue="TARGET CLASS")
    plt.tight_layout()
    output_path = os.path.join(results_dir, "knn_pairplot.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Pairplot saved to {output_path}")


def scale_features(df: pd.DataFrame, target_col: str = "TARGET CLASS"):
    """
    Standardize the feature columns.

    Returns
    -------
    X_scaled : DataFrame
        Scaled feature matrix.
    y : Series
        Target vector.
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    scaler = StandardScaler()
    scaler.fit(X)
    scaled_features = scaler.transform(X)

    X_scaled = pd.DataFrame(scaled_features, columns=X.columns)
    return X_scaled, y


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.3,
    random_state: int = 11
):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
    )


def evaluate_knn(
    X_train,
    X_test,
    y_train,
    y_test,
    n_neighbors: int = 5,
    label: str = "Model",
    results_dir: str = ".",
    filename: str | None = None,
):
    """
    Train and evaluate a KNN classifier for a given k.
    Saves the confusion matrix and classification report to a text file.
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)

    cm = confusion_matrix(y_test, pred)
    cr = classification_report(y_test, pred)
    acc = knn.score(X_test, y_test)

    # Print to console
    print(f"\n=== {label} (k = {n_neighbors}) ===")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(cr)

    # Save to text file
    if filename is None:
        safe_label = label.lower().replace(" ", "_")
        filename = f"{safe_label}_k{n_neighbors}.txt"

    output_path = os.path.join(results_dir, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{label} (k = {n_neighbors})\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Confusion matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification report:\n")
        f.write(cr)

    print(f"[INFO] Evaluation report saved to {output_path}")
    return knn, pred


def tune_k(
    X_train,
    X_test,
    y_train,
    y_test,
    k_min: int = 1,
    k_max: int = 40,
    results_dir: str = ".",
):
    """
    Compute error rate for KNN over a range of k values and plot the result.
    Saves the error plot into results_dir.

    Returns
    -------
    k_values : list[int]
    error_rate : list[float]
    best_k : int
    """
    error_rate = []
    k_values = list(range(k_min, k_max))

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        pred_k = knn.predict(X_test)
        error_rate.append(np.mean(pred_k != y_test))

    plt.figure(figsize=(8, 5))
    plt.plot(
        k_values,
        error_rate,
        linestyle="--",
        marker="o",
        markerfacecolor="red",
        markersize=6,
    )
    plt.xlabel("K")
    plt.ylabel("Error rate")
    plt.title("KNN error rate vs K")
    plt.tight_layout()

    output_path = os.path.join(results_dir, "knn_error_vs_k.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Error vs K plot saved to {output_path}")

    best_k_index = int(np.argmin(error_rate))
    best_k = k_values[best_k_index]
    print(f"[INFO] Approximate best k based on error rate: {best_k}")

    return k_values, error_rate, best_k


def main():
    data_path, results_dir = get_project_paths(results_dir_name="results")

    df = load_data(data_path)
    explore_data(df, results_dir=results_dir)
    X_scaled, y = scale_features(df, target_col="TARGET CLASS")
    X_train, X_test, y_train, y_test = train_test_split_data(
        X_scaled, y, test_size=0.3, random_state=11
    )

    evaluate_knn(
        X_train,
        X_test,
        y_train,
        y_test,
        n_neighbors=5,
        label="Baseline KNN model",
        results_dir=results_dir,
        filename="baseline_knn_k5.txt",
    )

    _, _, best_k = tune_k(
        X_train,
        X_test,
        y_train,
        y_test,
        k_min=1,
        k_max=40,
        results_dir=results_dir,
    )

    final_k = 30
    evaluate_knn(
        X_train,
        X_test,
        y_train,
        y_test,
        n_neighbors=final_k,
        label="Final KNN model",
        results_dir=results_dir,
        filename=f"final_knn_k{final_k}.txt",
    )

if __name__ == "__main__":
    main()
