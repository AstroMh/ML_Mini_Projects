import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

def get_project_paths(results_dir_name: str = "results"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, results_dir_name)
    os.makedirs(results_dir, exist_ok=True)
    return base_dir, results_dir


def load_data(as_frame: bool = True) -> pd.DataFrame:
    housing = fetch_california_housing(as_frame=as_frame)
    df = housing.frame  # includes feature columns + 'MedHouseVal' target
    return df

def explore_data(df: pd.DataFrame, results_dir: str) -> None:
    # Subset for pairplot to keep it readable
    cols_for_pairplot = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveOccup",
        "MedHouseVal",
    ]
    subset = df[cols_for_pairplot]

    pairplot_path = os.path.join(results_dir, "california_pairplot.png")
    sns.pairplot(subset)
    plt.tight_layout()
    plt.savefig(pairplot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Pairplot saved to {pairplot_path}")

    corr = df.corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("California Housing – Correlation Heatmap")
    plt.tight_layout()
    heatmap_path = os.path.join(results_dir, "california_corr_heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Correlation heatmap saved to {heatmap_path}")


def prepare_features(df: pd.DataFrame):
    """
    Prepare feature matrix X and target vector y.

    Features (from sklearn docs):
      - MedInc: median income in block group
      - HouseAge: median house age
      - AveRooms: average number of rooms
      - AveBedrms: average number of bedrooms
      - Population: block group population
      - AveOccup: average number of household members
      - Latitude: block group latitude
      - Longitude: block group longitude

    Target:
      - MedHouseVal: median house value (in 100,000s USD)
    """
    feature_cols = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]
    target_col = "MedHouseVal"

    X = df[feature_cols]
    y = df[target_col]
    return X, y, feature_cols


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
    )

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_regression(
    model,
    X_test,
    y_test,
    results_dir: str,
    filename: str | None = None,
):
    """
    Evaluate a regression model and save metrics to a text file.
    Computes MAE, MSE, RMSE, R^2.
    """
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n=== California Housing – Linear Regression Evaluation ===")
    print(f"MAE : {mae:.4f}")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2 : {r2:.4f}")

    if filename is None:
        filename = "california_linear_regression_report.txt"

    report_path = os.path.join(results_dir, filename)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("California Housing – Linear Regression Evaluation\n")
        f.write(f"MAE : {mae:.6f}\n")
        f.write(f"MSE : {mse:.6f}\n")
        f.write(f"RMSE: {rmse:.6f}\n")
        f.write(f"R^2 : {r2:.6f}\n")

    print(f"[INFO] Evaluation report saved to {report_path}")

    return y_pred, {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


def plot_predictions(
    y_test,
    y_pred,
    results_dir: str,
    filename: str = "california_pred_vs_actual.png",
):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_pred, y_test, alpha=0.5)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("Predicted MedHouseVal")
    plt.ylabel("Actual MedHouseVal")
    plt.title("Predicted vs Actual – California Housing")
    plt.tight_layout()

    scatter_path = os.path.join(results_dir, filename)
    plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Predicted vs actual plot saved to {scatter_path}")


def plot_residuals(
    y_test,
    y_pred,
    results_dir: str,
    filename: str = "california_residuals.png",
):
    residuals = y_test - y_pred

    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=50, kde=True)
    plt.xlabel("Residuals (Actual - Predicted)")
    plt.title("Residuals Distribution – California Housing")
    plt.tight_layout()

    residuals_path = os.path.join(results_dir, filename)
    plt.savefig(residuals_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Residuals plot saved to {residuals_path}")


def plot_coefficients(
    model,
    feature_names,
    results_dir: str,
    filename: str = "california_coefficients.png",
):
    coeffs = pd.Series(model.coef_, index=feature_names).sort_values()

    plt.figure(figsize=(8, 6))
    coeffs.plot(kind="barh")
    plt.xlabel("Coefficient value")
    plt.title("Linear Regression Coefficients – California Housing")
    plt.tight_layout()

    coeffs_path = os.path.join(results_dir, filename)
    plt.savefig(coeffs_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Coefficients plot saved to {coeffs_path}")


def save_coefficients_table(
    model,
    feature_names,
    results_dir: str,
    filename_csv: str = "california_coefficients.csv",
    filename_txt: str = "california_coefficients.txt",
):
    coeffs = pd.DataFrame(
        {"feature": feature_names, "coefficient": model.coef_}
    )

    csv_path = os.path.join(results_dir, filename_csv)
    coeffs.to_csv(csv_path, index=False)

    txt_path = os.path.join(results_dir, filename_txt)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("California Housing – Linear Regression Coefficients\n\n")
        for feat, coef in zip(feature_names, model.coef_):
            f.write(f"{feat}: {coef:.6f}\n")

    print(f"[INFO] Coefficients saved to {csv_path} and {txt_path}")

def main():
    _, results_dir = get_project_paths(results_dir_name="results")

    df = load_data(as_frame=True)
    explore_data(df, results_dir=results_dir)
    X, y, feature_cols = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=0.2, random_state=42
    )
    model = train_linear_regression(X_train, y_train)

    y_pred, metrics = evaluate_regression(
        model,
        X_test,
        y_test,
        results_dir=results_dir,
        filename="california_linear_regression_report.txt",
    )

    plot_predictions(y_test, y_pred, results_dir=results_dir)
    plot_residuals(y_test, y_pred, results_dir=results_dir)

    plot_coefficients(model, feature_cols, results_dir=results_dir)
    save_coefficients_table(model, feature_cols, results_dir=results_dir)

if __name__ == "__main__":
    main()
