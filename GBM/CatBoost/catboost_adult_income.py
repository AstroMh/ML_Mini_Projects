import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from catboost import CatBoostClassifier


def make_dirs(base_dir="results"):
    plots_dir = os.path.join(base_dir, "plots")
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    return base_dir, plots_dir, reports_dir


def load_adult():
    adult = fetch_openml(name="adult", version=2, as_frame=True)
    X = adult.data.copy()
    y = adult.target.copy()
    return X, y


def clean_adult(X, y):
    X = X.replace("?", np.nan)
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].str.strip()
    data = X.copy()
    data["target"] = y
    data = data.dropna()
    X = data.drop(columns=["target"])
    y = data["target"]
    return X, y


def get_col_types(X):
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    return cat_cols, num_cols


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def save_bar(series, title, filepath, top_n=None):
    s = series.copy()
    if top_n is not None:
        s = s.head(top_n)
    s.plot(kind="bar")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def save_hist(series, title, filepath, bins=30):
    series.hist(bins=bins)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def save_corr_heatmap(df, filepath, title):
    corr = df.corr()
    plt.figure(figsize=(8, 6))
    plt.imshow(corr, aspect="auto")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def save_cm_plot(cm, title, filepath):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title(title)
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def save_text(text, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)


def compute_metrics(y_true, y_pred, pos_label, y_proba=None):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=pos_label),
        "recall": recall_score(y_true, y_pred, pos_label=pos_label),
        "f1": f1_score(y_true, y_pred, pos_label=pos_label),
    }
    if y_proba is not None:
        y_true_bin = (y_true == pos_label).astype(int)
        out["roc_auc"] = roc_auc_score(y_true_bin, y_proba)
    return out


def run_eda(X, y, cat_cols, num_cols, results_dir, plots_dir):
    y.value_counts().to_csv(os.path.join(results_dir, "target_counts.csv"), header=["count"])
    save_bar(y.value_counts(), "Target distribution", os.path.join(plots_dir, "target_distribution.png"))

    missing_counts = X.isna().sum().sort_values(ascending=False)
    missing_counts.to_csv(os.path.join(results_dir, "missing_values_by_column.csv"), header=["missing_count"])

    for col in num_cols[:3]:
        save_hist(X[col], f"Histogram: {col}", os.path.join(plots_dir, f"hist_{col}.png"))

    if len(num_cols) > 1:
        corr_cols = num_cols[:10]
        save_corr_heatmap(
            X[corr_cols],
            os.path.join(plots_dir, "correlation_heatmap_numeric_subset.png"),
            "Correlation heatmap (numeric subset)"
        )

    for col in cat_cols[:2]:
        vc = X[col].value_counts().head(10)
        save_bar(vc, f"Top categories: {col}", os.path.join(plots_dir, f"cat_{col}.png"))


def train_baseline_decision_tree(X_train, y_train, cat_cols, num_cols, random_state=42):
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )
    model = Pipeline(steps=[
        ("prep", preprocess),
        ("model", DecisionTreeClassifier(random_state=random_state, max_depth=8))
    ])
    model.fit(X_train, y_train)
    return model


def train_catboost(X_train, y_train, X_test, y_test, cat_cols, random_seed=42):
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=random_seed,
        verbose=100
    )
    model.fit(
        X_train, y_train,
        cat_features=cat_cols,
        eval_set=(X_test, y_test),
        use_best_model=True
    )
    return model


def save_model_outputs(
    name, y_test, y_pred, pos_label, plots_dir, reports_dir, results_dir, y_proba=None
):
    cm = confusion_matrix(y_test, y_pred)
    save_cm_plot(cm, f"Confusion Matrix - {name}", os.path.join(plots_dir, f"confusion_matrix_{name}.png"))
    np.savetxt(os.path.join(results_dir, f"confusion_matrix_{name}.csv"), cm, delimiter=",", fmt="%d")

    report = classification_report(y_test, y_pred)
    save_text(report, os.path.join(reports_dir, f"classification_report_{name}.txt"))

    metrics = compute_metrics(y_test, y_pred, pos_label=pos_label, y_proba=y_proba)
    pd.Series(metrics).to_csv(os.path.join(results_dir, f"metrics_{name}.csv"), header=["value"])
    return metrics


def save_feature_importance_catboost(model, X_columns, plots_dir, results_dir):
    importances = model.get_feature_importance()
    fi = pd.Series(importances, index=X_columns).sort_values(ascending=False).head(15)
    fi.sort_values().plot(kind="barh")
    plt.title("CatBoost Feature Importance (Top 15)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "feature_importances_catboost.png"))
    plt.close()
    fi.to_csv(os.path.join(results_dir, "feature_importances_top15.csv"), header=["importance"])


def main():
    results_dir, plots_dir, reports_dir = make_dirs("results")
    POS_LABEL = ">50K"

    X_raw, y_raw = load_adult()
    X, y = clean_adult(X_raw, y_raw)
    cat_cols, num_cols = get_col_types(X)

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    run_eda(X, y, cat_cols, num_cols, results_dir, plots_dir)

    baseline = train_baseline_decision_tree(X_train, y_train, cat_cols, num_cols, random_state=42)
    y_pred_base = baseline.predict(X_test)
    metrics_base = save_model_outputs(
        name="baseline",
        y_test=y_test,
        y_pred=y_pred_base,
        pos_label=POS_LABEL,
        plots_dir=plots_dir,
        reports_dir=reports_dir,
        results_dir=results_dir
    )

    cb = train_catboost(X_train, y_train, X_test, y_test, cat_cols, random_seed=42)
    y_pred_cb = cb.predict(X_test)
    y_proba_cb = cb.predict_proba(X_test)[:, 1]
    metrics_cb = save_model_outputs(
        name="catboost",
        y_test=y_test,
        y_pred=y_pred_cb,
        pos_label=POS_LABEL,
        plots_dir=plots_dir,
        reports_dir=reports_dir,
        results_dir=results_dir,
        y_proba=y_proba_cb
    )

    compare = pd.DataFrame([metrics_base, metrics_cb], index=["baseline_decision_tree", "catboost"])
    compare.to_csv(os.path.join(results_dir, "metrics_comparison.csv"), index=True)

    try:
        save_feature_importance_catboost(cb, X.columns, plots_dir, results_dir)
    except Exception as e:
        print("Feature importance skipped:", e)

    print("Done. Artifacts saved in:", os.path.abspath(results_dir))


main()
