import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

def get_project_paths(results_dir_name: str = "results"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, results_dir_name)
    os.makedirs(results_dir, exist_ok=True)
    return base_dir, results_dir

def load_data():
    digits = load_digits()
    X = digits.data
    y = digits.target
    images = digits.images
    return X, y, images

def plot_sample_digits(images, labels, results_dir: str, filename: str = "digits_sample_grid.png", n: int = 25):
    n = min(n, len(images))
    side = int(np.ceil(np.sqrt(n)))

    fig, axes = plt.subplots(side, side, figsize=(6, 6))
    axes = axes.ravel()

    for i in range(side * side):
        axes[i].axis("off")
        if i < n:
            axes[i].imshow(images[i], cmap="gray_r")
            axes[i].set_title(str(labels[i]), fontsize=8)

    plt.tight_layout()
    output_path = os.path.join(results_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Sample digits grid saved to {output_path}")


def plot_example_predictions(images, y_true, y_pred, results_dir: str,
                             filename: str = "digits_sample_predictions.png", n: int = 25):
    n = min(n, len(images))
    side = int(np.ceil(np.sqrt(n)))

    fig, axes = plt.subplots(side, side, figsize=(7, 7))
    axes = axes.ravel()

    for i in range(side * side):
        axes[i].axis("off")
        if i < n:
            axes[i].imshow(images[i], cmap="gray_r")
            pred = y_pred[i]
            true = y_true[i]
            correct = pred == true
            title = f"p:{pred} / t:{true}"
            color = "green" if correct else "red"
            axes[i].set_title(title, fontsize=8, color=color)

    plt.tight_layout()
    output_path = os.path.join(results_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Sample predictions grid saved to {output_path}")

def evaluate_knn(
    X_train,
    X_test,
    y_train,
    y_test,
    n_neighbors: int = 5,
    label: str = "KNN digits model",
    results_dir: str = ".",
    filename: str | None = None,
    cm_filename: str | None = None,
):
    """
    Train and evaluate a KNN classifier for a given k.
    Saves:
      - Text report (accuracy + classification report)
      - Confusion matrix heatmap
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    acc = knn.score(X_test, y_test)

    print(f"\n=== {label} (k = {n_neighbors}) ===")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(cr)

    if filename is None:
        safe_label = label.lower().replace(" ", "_")
        filename = f"{safe_label}_k{n_neighbors}.txt"

    report_path = os.path.join(results_dir, filename)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"{label} (k = {n_neighbors})\n")
        f.write(f"Accuracy: {acc:.6f}\n\n")
        f.write("Confusion matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification report:\n")
        f.write(cr)

    print(f"[INFO] Evaluation report saved to {report_path}")

    if cm_filename is None:
        cm_filename = f"digits_confusion_matrix_k{n_neighbors}.png"

    cm_path = os.path.join(results_dir, cm_filename)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion Matrix (k = {n_neighbors})")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix heatmap saved to {cm_path}")

    return knn, y_pred


def tune_k(
    X_train,
    X_test,
    y_train,
    y_test,
    k_min: int = 1,
    k_max: int = 20,
    results_dir: str = ".",
    filename: str = "digits_error_vs_k.png",
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
    plt.title("KNN error rate vs K (digits dataset)")
    plt.tight_layout()

    output_path = os.path.join(results_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Error vs K plot saved to {output_path}")

    best_k_index = int(np.argmin(error_rate))
    best_k = k_values[best_k_index]
    print(f"[INFO] Approximate best k based on error rate: {best_k}")

    return k_values, error_rate, best_k

def main():
    _, results_dir = get_project_paths(results_dir_name="results")

    X, y, images = load_data()
    plot_sample_digits(images, y, results_dir=results_dir,
                       filename="digits_sample_grid.png", n=25)

    X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
        X,
        y,
        images,
        test_size=0.3,
        random_state=11,
        stratify=y,
    )

    baseline_k = 5
    _, y_pred_baseline = evaluate_knn(
        X_train,
        X_test,
        y_train,
        y_test,
        n_neighbors=baseline_k,
        label="Baseline KNN digits model",
        results_dir=results_dir,
        filename=f"baseline_knn_digits_k{baseline_k}.txt",
        cm_filename=f"digits_confusion_matrix_k{baseline_k}.png",
    )

    _, _, best_k = tune_k(
        X_train,
        X_test,
        y_train,
        y_test,
        k_min=1,
        k_max=20,
        results_dir=results_dir,
        filename="digits_error_vs_k.png",
    )

    final_k = best_k
    model, y_pred_final = evaluate_knn(
        X_train,
        X_test,
        y_train,
        y_test,
        n_neighbors=final_k,
        label="Final KNN digits model",
        results_dir=results_dir,
        filename=f"final_knn_digits_k{final_k}.txt",
        cm_filename=f"digits_confusion_matrix_k{final_k}.png",
    )

    plot_example_predictions(
        img_test,
        y_test,
        y_pred_final,
        results_dir=results_dir,
        filename=f"digits_sample_predictions_k{final_k}.png",
        n=25,
    )

if __name__ == "__main__":
    main()
