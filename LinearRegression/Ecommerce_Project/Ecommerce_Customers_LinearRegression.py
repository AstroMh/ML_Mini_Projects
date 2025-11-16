import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

def get_project_paths(results_dir_name: str = "results"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "Ecommerce Customers")
    results_dir = os.path.join(base_dir, results_dir_name)
    os.makedirs(results_dir, exist_ok=True)
    return data_path, results_dir


def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    # Drop columns that are not useful for the regression model
    for col in ["Address", "Email", "Avatar"]:
        if col in df.columns:
            df = df.drop(col, axis=1)
    return df

def explore_data(df: pd.DataFrame, results_dir: str) -> None:
    # Pairplot
    pairplot_path = os.path.join(results_dir, "linear_regression_pairplot.png")
    sns.pairplot(df)
    plt.tight_layout()
    plt.savefig(pairplot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Pairplot saved to {pairplot_path}")

    # Correlation heatmap
    corr = df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    heatmap_path = os.path.join(results_dir, "linear_regression_corr_heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Correlation heatmap saved to {heatmap_path}")


def prepare_features(df: pd.DataFrame):
    feature_cols = [
        "Avg. Session Length",
        "Time on App",
        "Time on Website",
        "Length of Membership",
    ]
    target_col = "Yearly Amount Spent"

    X = df[feature_cols]
    y = df[target_col]
    return X, y


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.3,
    random_state: int = 11,
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
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Print to console
    print("\n=== Linear Regression Evaluation ===")
    print(f"MAE : {mae:.4f}")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Save to text file
    if filename is None:
        filename = "linear_regression_report.txt"

    report_path = os.path.join(results_dir, filename)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Linear Regression Evaluation\n")
        f.write(f"MAE : {mae:.6f}\n")
        f.write(f"MSE : {mse:.6f}\n")
        f.write(f"RMSE: {rmse:.6f}\n")

    print(f"[INFO] Evaluation report saved to {report_path}")

    return y_pred, {"mae": mae, "mse": mse, "rmse": rmse}


def plot_predictions(
    y_test,
    y_pred,
    results_dir: str,
):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_pred, y_test, alpha=0.6)
    plt.xlabel("Predicted Yearly Amount Spent")
    plt.ylabel("Actual Yearly Amount Spent")
    plt.title("Predicted vs Actual")
    plt.tight_layout()

    scatter_path = os.path.join(results_dir, "linear_regression_pred_vs_actual.png")
    plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Predicted vs actual plot saved to {scatter_path}")


def plot_residuals(
    y_test,
    y_pred,
    results_dir: str,
):
    residuals = y_test - y_pred

    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=50, kde=True)
    plt.xlabel("Residuals (Actual - Predicted)")
    plt.title("Residuals Distribution")
    plt.tight_layout()

    residuals_path = os.path.join(results_dir, "linear_regression_residuals.png")
    plt.savefig(residuals_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Residuals plot saved to {residuals_path}")


def save_coefficients(
    model,
    feature_names,
    results_dir: str,
    filename_csv: str = "linear_regression_coefficients.csv",
    filename_txt: str = "linear_regression_coefficients.txt",
):
    """
    Save model coefficients to CSV and TXT files.
    """
    coeffs = pd.DataFrame(
        {"feature": feature_names, "coefficient": model.coef_}
    )

    csv_path = os.path.join(results_dir, filename_csv)
    coeffs.to_csv(csv_path, index=False)

    txt_path = os.path.join(results_dir, filename_txt)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Linear Regression Coefficients\n\n")
        for feat, coef in zip(feature_names, model.coef_):
            f.write(f"{feat}: {coef:.6f}\n")

    print(f"[INFO] Coefficients saved to {csv_path} and {txt_path}")

def main():
    data_path, results_dir = get_project_paths(results_dir_name="results")

    df = load_data(data_path)
    explore_data(df, results_dir=results_dir)
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=0.3, random_state=11
    )
    model = train_linear_regression(X_train, y_train)

    y_pred, metrics = evaluate_regression(
        model,
        X_test,
        y_test,
        results_dir=results_dir,
        filename="linear_regression_report.txt",
    )

    plot_predictions(y_test, y_pred, results_dir=results_dir)
    plot_residuals(y_test, y_pred, results_dir=results_dir)

    feature_cols = [
        "Avg. Session Length",
        "Time on App",
        "Time on Website",
        "Length of Membership",
    ]
    save_coefficients(model, feature_cols, results_dir=results_dir)

if __name__ == "__main__":
    main()
