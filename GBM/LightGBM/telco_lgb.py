import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

import lightgbm as lgb


def paths():
    base = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base, "results")
    plots_dir = os.path.join(results_dir, "plots")
    reports_dir = os.path.join(results_dir, "reports")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    return base, results_dir, plots_dir, reports_dir


def load_telco(csv_path):
    df = pd.read_csv(csv_path)
    df = df.copy()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    return df


def split_features(df):
    X = df.drop(columns=["Churn"])
    y = df["Churn"].astype(int)
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = [c for c in X.columns if c not in categorical_cols]
    for c in categorical_cols:
        X[c] = X[c].astype("category")
    return X, y, categorical_cols, numerical_cols


def split_train_test(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def save_plot(path):
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_target_distribution(y, plots_dir):
    y.value_counts().plot(kind="bar")
    plt.title("Churn distribution")
    save_plot(os.path.join(plots_dir, "churn_distribution.png"))


def plot_box_by_target(df, col, plots_dir, filename):
    df.boxplot(column=col, by="Churn")
    plt.title(f"{col} vs Churn")
    plt.suptitle("")
    save_plot(os.path.join(plots_dir, filename))


def plot_crosstab_stacked(df, col, plots_dir, filename):
    pd.crosstab(df[col], df["Churn"], normalize="index").plot(kind="bar", stacked=True)
    plt.title(f"Churn rate by {col}")
    save_plot(os.path.join(plots_dir, filename))


def run_min_eda(df, plots_dir):
    plot_target_distribution(df["Churn"], plots_dir)
    if "tenure" in df.columns:
        plot_box_by_target(df, "tenure", plots_dir, "tenure_vs_churn.png")
    if "MonthlyCharges" in df.columns:
        plot_box_by_target(df, "MonthlyCharges", plots_dir, "monthly_charges_vs_churn.png")
    if "Contract" in df.columns:
        plot_crosstab_stacked(df, "Contract", plots_dir, "contract_vs_churn.png")


def build_logreg_baseline(categorical_cols, numerical_cols, random_state=42):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numerical_cols),
        ]
    )
    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", LogisticRegression(max_iter=2000, random_state=random_state))
        ]
    )
    return model


def lgb_params(y_train, seed=42):
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    spw = (neg / max(pos, 1))

    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l2": 1.0,
        "scale_pos_weight": spw,
        "verbosity": -1,
        "seed": seed,
    }


def train_lightgbm(X_train, y_train, X_valid, y_valid, categorical_cols, params):
    dtrain = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=categorical_cols,
        free_raw_data=False
    )
    dvalid = lgb.Dataset(
        X_valid,
        label=y_valid,
        categorical_feature=categorical_cols,
        free_raw_data=False
    )
    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100),
    ]
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=callbacks
    )
    return model


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def save_confusion(y_true, y_pred, results_dir, plots_dir, name):
    cm = confusion_matrix(y_true, y_pred)
    np.savetxt(os.path.join(results_dir, f"confusion_matrix_{name}.csv"), cm, delimiter=",", fmt="%d")
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, aspect="auto")
    plt.title(f"Confusion Matrix - {name}")
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    save_plot(os.path.join(plots_dir, f"confusion_matrix_{name}.png"))


def save_report(y_true, y_pred, reports_dir, name):
    rep = classification_report(y_true, y_pred, digits=4, zero_division=0)
    with open(os.path.join(reports_dir, f"classification_report_{name}.txt"), "w", encoding="utf-8") as f:
        f.write(rep)


def save_metrics(metrics, results_dir, name):
    pd.Series(metrics).to_csv(os.path.join(results_dir, f"metrics_{name}.csv"), header=["value"])


def save_lgb_importance(model, feature_names, plots_dir, results_dir, top_n=15):
    imp = pd.Series(model.feature_importance(importance_type="gain"), index=feature_names).sort_values(ascending=False)
    imp.head(top_n).to_csv(os.path.join(results_dir, "feature_importance_top.csv"), header=["gain"])
    imp.head(top_n).sort_values().plot(kind="barh")
    plt.title(f"LightGBM Feature Importance (Top {top_n})")
    save_plot(os.path.join(plots_dir, "feature_importance_lightgbm.png"))


def main():
    base, results_dir, plots_dir, reports_dir = paths()
    csv_path = os.path.join(base, "data", "telco_churn.csv")

    df = load_telco(csv_path)
    run_min_eda(df, plots_dir)

    X, y, categorical_cols, numerical_cols = split_features(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2, random_state=42)

    baseline = build_logreg_baseline(categorical_cols, numerical_cols, random_state=42)
    baseline.fit(X_train, y_train)
    y_proba_base = baseline.predict_proba(X_test)[:, 1]
    y_pred_base = (y_proba_base >= 0.5).astype(int)

    metrics_base = compute_metrics(y_test, y_pred_base, y_proba_base)
    save_confusion(y_test, y_pred_base, results_dir, plots_dir, "baseline_logreg")
    save_report(y_test, y_pred_base, reports_dir, "baseline_logreg")
    save_metrics(metrics_base, results_dir, "baseline_logreg")

    params = lgb_params(y_train, seed=42)
    lgb_model = train_lightgbm(X_train, y_train, X_test, y_test, categorical_cols, params)
    y_proba_lgb = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    y_pred_lgb = (y_proba_lgb >= 0.5).astype(int)

    metrics_lgb = compute_metrics(y_test, y_pred_lgb, y_proba_lgb)
    save_confusion(y_test, y_pred_lgb, results_dir, plots_dir, "lightgbm")
    save_report(y_test, y_pred_lgb, reports_dir, "lightgbm")
    save_metrics(metrics_lgb, results_dir, "lightgbm")
    save_lgb_importance(lgb_model, X_train.columns, plots_dir, results_dir, top_n=15)

    comparison = pd.DataFrame([metrics_base, metrics_lgb], index=["baseline_logreg", "lightgbm"])
    comparison.to_csv(os.path.join(results_dir, "metrics_comparison.csv"), index=True)


if __name__ == "__main__":
    main()
