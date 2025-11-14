import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report


def load_data(path: str = "KNN_Project_Data") -> pd.DataFrame:
    """
    Load the KNN project dataset.

    Parameters
    ----------
    path : str
        Path to the CSV file (as used in the original notebook).

    Returns
    -------
    DataFrame
        Loaded dataset.
    """
    df = pd.read_csv(path, index_col=0)
    return df


def explore_data(df: pd.DataFrame) -> None:
    """
    Basic exploratory visualization: pairplot colored by TARGET CLASS.
    """
    sns.pairplot(df, hue="TARGET CLASS")
    plt.tight_layout()
    plt.show()


def scale_features(df: pd.DataFrame, target_col: str = "TARGET CLASS"):
    """
    Standardize the feature columns.

    Parameters
    ----------
    df : DataFrame
        Original dataset including the target column.
    target_col : str
        Name of the target column.

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


def train_test_split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.3, random_state: int = 11):
    """
    Split features and target into train and test sets.
    """
    return train_test_split(X, y, test_size=test_size, shuffle=True, random_state=random_state)


def evaluate_knn(X_train, X_test, y_train, y_test, n_neighbors: int = 5, label: str = "Model"):
    """
    Train and evaluate a KNN classifier for a given k.
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)

    print(f"\n=== {label} (k = {n_neighbors}) ===")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, pred))
    print("\nClassification report:")
    print(classification_report(y_test, pred))

    return knn, pred


def tune_k(X_train, X_test, y_train, y_test, k_min: int = 1, k_max: int = 40):
    """
    Compute error rate for KNN over a range of k values and plot the result.

    Returns
    -------
    k_values : list[int]
    error_rate : list[float]
    """
    error_rate = []
    k_values = list(range(k_min, k_max))

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        pred_k = knn.predict(X_test)
        error_rate.append(np.mean(pred_k != y_test))

    # Plot error vs k
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, error_rate, linestyle="--", marker="o", markerfacecolor="red")
    plt.xlabel("K")
    plt.ylabel("Error rate")
    plt.title("KNN error rate vs K")
    plt.tight_layout()
    plt.show()

    # Best k (smallest error)
    best_k_index = int(np.argmin(error_rate))
    best_k = k_values[best_k_index]
    print(f"\nApproximate best k based on error rate: {best_k}")

    return k_values, error_rate, best_k


def main():
    # 1. Load data
    df = load_data("KNN_Project_Data")

    # 2. Optional: exploratory pairplot
    explore_data(df)

    # 3. Scale features
    X_scaled, y = scale_features(df, target_col="TARGET CLASS")

    # 4. Train/test split
    X_train, X_test, y_train, y_test = train_test_split_data(X_scaled, y, test_size=0.3, random_state=11)

    # 5. Baseline KNN model (sklearn's default k)
    baseline_knn, _ = evaluate_knn(X_train, X_test, y_train, y_test,
                                   n_neighbors=5,
                                   label="Baseline KNN model")

    # 6. Hyperparameter tuning over k = 1..39
    _, _, best_k = tune_k(X_train, X_test, y_train, y_test, k_min=1, k_max=40)

    # 7. Final model – using your original choice k=30
    final_k = 30
    final_knn, _ = evaluate_knn(X_train, X_test, y_train, y_test,
                                n_neighbors=final_k,
                                label="Final KNN model")


if __name__ == "__main__":
    main()
