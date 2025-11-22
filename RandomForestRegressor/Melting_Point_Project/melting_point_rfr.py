# ================================
# PolyML: mpC prediction script
# ================================

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print(train_df.head())


def smiles_basic_features(s: str) -> dict:
    """String-based SMILES heuristics that don't rely on RDKit."""
    if not isinstance(s, str):
        s = "" if s is np.nan else str(s)

    feat = {
        "length": len(s),
        "num_C": s.count("C"),
        "num_c_aromatic": s.count("c"),
        "num_N": s.count("N"),
        "num_n_aromatic": s.count("n"),
        "num_O": s.count("O"),
        "num_o_aromatic": s.count("o"),
        "num_H": s.count("H"),
        "num_S": s.count("S"),
        "num_F": s.count("F"),
        "num_Cl": s.count("Cl"),
        "num_Br": s.count("Br"),
        "num_I": s.count("I"),
        "num_double": s.count("="),
        "num_triple": s.count("#"),
        "num_branches": s.count("("),
        "num_rings": sum(ch.isdigit() for ch in s),
        "num_plus": s.count("+"),
        "num_minus": s.count("-"),
        "num_slash": s.count("/"),
        "num_backslash": s.count("\\"),
        "num_aromatic_chars": sum(s.count(ch) for ch in "cnosp"),
    }
    feat["frac_aromatic"] = feat["num_aromatic_chars"] / (feat["length"] + 1e-6)

    # simple interaction-style features
    feat["ring_density"] = feat["num_rings"] / (feat["length"] + 1e-6)
    feat["branch_density"] = feat["num_branches"] / (feat["length"] + 1e-6)
    feat["hetero_atoms"] = (
        feat["num_N"]
        + feat["num_O"]
        + feat["num_S"]
        + feat["num_F"]
        + feat["num_Cl"]
        + feat["num_Br"]
        + feat["num_I"]
    )

    return feat


# Try to use RDKit descriptors if available
use_rdkit = False
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    use_rdkit = True
    print("RDKit available: will add RDKit descriptors.")
except Exception:
    print("RDKit NOT available, using only SMILES string features.")
    use_rdkit = False


def rdkit_features(smiles: str) -> dict:
    """Compute a handful of RDKit descriptors; return empty dict if invalid or RDKit off."""
    if not use_rdkit:
        return {}
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        return {
            "MolWt": Descriptors.MolWt(mol),
            "MolLogP": Crippen.MolLogP(mol),
            "TPSA": rdMolDescriptors.CalcTPSA(mol),
            "NumHDonors": rdMolDescriptors.CalcNumHBD(mol),
            "NumHAcceptors": rdMolDescriptors.CalcNumHBA(mol),
            "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "HeavyAtomCount": rdMolDescriptors.CalcNumHeavyAtoms(mol),
            "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        }
    except Exception:
        return {}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Combine basic SMILES string features and (optionally) RDKit descriptors."""
    basic = df["smiles"].apply(smiles_basic_features).apply(pd.Series)
    if use_rdkit:
        rd = df["smiles"].apply(rdkit_features).apply(pd.Series)
        feats = pd.concat([basic, rd], axis=1)
    else:
        feats = basic
    return feats

def explore_raw_data(train_df: pd.DataFrame, results_dir: str = RESULTS_DIR) -> None:
    """Simple visualizations directly on train_df."""
    sns.set(style="whitegrid")

    # Histogram of target mpC
    plt.figure(figsize=(8, 5))
    sns.histplot(train_df["mpC"], bins=50, kde=True)
    plt.xlabel("mpC (°C)")
    plt.title("Distribution of melting point (mpC)")
    plt.tight_layout()
    path = os.path.join(results_dir, "mpC_histogram.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved {path}")

    # Distribution of SMILES length
    smiles_len = train_df["smiles"].astype(str).str.len()
    plt.figure(figsize=(8, 5))
    sns.histplot(smiles_len, bins=50, kde=True)
    plt.xlabel("SMILES length")
    plt.title("Distribution of SMILES length")
    plt.tight_layout()
    path = os.path.join(results_dir, "smiles_length_histogram.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved {path}")

    # Jointplot: mpC vs SMILES length
    tmp = pd.DataFrame({"mpC": train_df["mpC"], "smiles_length": smiles_len})
    jp = sns.jointplot(
        data=tmp,
        x="smiles_length",
        y="mpC",
        kind="hex",
        height=6,
    )
    jp.fig.suptitle("mpC vs SMILES length", y=1.02)
    path = os.path.join(results_dir, "mpC_vs_smiles_length_jointplot.png")
    jp.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(jp.fig)
    print(f"[INFO] Saved {path}")


print("Building features...")
train_features = build_features(train_df)
test_features = build_features(test_df)
print("Train features shape:", train_features.shape)
print("Test features shape:", test_features.shape)

common_cols = train_features.columns.intersection(test_features.columns)
train_features = train_features[common_cols]
test_features = test_features[common_cols]

median_vals = train_features.median()
train_features = train_features.fillna(median_vals)
test_features = test_features.fillna(median_vals)

X = train_features.values
y = train_df["mpC"].values
X_test = test_features.values

print("Final feature matrix shape:", X.shape)

def explore_feature_correlations(
    train_features: pd.DataFrame,
    target: pd.Series,
    results_dir: str = RESULTS_DIR,
    top_n: int = 20,
) -> None:
    """
    Compute correlations between features and target,
    plot:
      - barplot of top |corr|
      - heatmap of correlation matrix of top N features
    """
    df_corr = train_features.copy()
    df_corr["mpC"] = target.values

    corr_with_target = df_corr.corr()["mpC"].drop("mpC").sort_values(key=lambda s: s.abs(), ascending=False)

    top_corr = corr_with_target.head(top_n)
    plt.figure(figsize=(8, max(4, top_n * 0.3)))
    sns.barplot(
        x=top_corr.values,
        y=top_corr.index,
        orient="h",
    )
    plt.xlabel("Correlation with mpC")
    plt.title(f"Top {top_n} features correlated with mpC")
    plt.tight_layout()
    path = os.path.join(results_dir, "top_feature_correlations_with_mpC.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved {path}")

    # Heatmap of correlations among these top features + mpC
    cols = list(top_corr.index) + ["mpC"]
    corr_mat = df_corr[cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_mat, annot=False, cmap="coolwarm", center=0)
    plt.title("Correlation heatmap of top features and mpC")
    plt.tight_layout()
    path = os.path.join(results_dir, "top_features_correlation_heatmap.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved {path}")

print("Running basic EDA and feature correlation plots...")
explore_raw_data(train_df, RESULTS_DIR)
explore_feature_correlations(train_features, train_df["mpC"], RESULTS_DIR, top_n=20)


X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=11
)

baseline_pred = np.full_like(y_val, y_tr.mean(), dtype=float)
baseline_mae = mean_absolute_error(y_val, baseline_pred)
print("Baseline MAE (predict mean):", baseline_mae)

rf_baseline = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    random_state=101,
    n_jobs=-1,
)
rf_baseline.fit(X_tr, y_tr)
val_pred_rf = rf_baseline.predict(X_val)
rf_mae = mean_absolute_error(y_val, val_pred_rf)
print("Baseline RandomForest MAE:", rf_mae)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
rf_cv_maes = -cross_val_score(
    rf_baseline,
    X,
    y,
    cv=kf,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
)
print("RF CV MAE mean:", rf_cv_maes.mean(), "±", rf_cv_maes.std())


# --------------------------------
# XGBoost or HistGradientBoosting
# --------------------------------
use_xgb = False
try:
    from xgboost import XGBRegressor

    use_xgb = True
    print("Using XGBoostRegressor as main model.")
except Exception:
    print("xgboost not available, using HistGradientBoostingRegressor instead.")
    from sklearn.ensemble import HistGradientBoostingRegressor

if use_xgb:
    base_model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )

    base_maes = -cross_val_score(
        base_model,
        X,
        y,
        cv=kf,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    print("XGB base CV MAE mean:", base_maes.mean(), "±", base_maes.std())

    DO_TUNING = True
    if DO_TUNING:
        print("Running RandomizedSearchCV for XGBRegressor...")
        param_dist = {
            "n_estimators": [400, 800, 1200],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.02, 0.03, 0.05, 0.08],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "min_child_weight": [1, 3, 5],
        }

        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_dist,
            n_iter=25,
            cv=3,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            random_state=42,
            verbose=1,
        )

        search.fit(X, y)
        print("Best XGB CV MAE:", -search.best_score_)
        print("Best params:", search.best_params_)
        best_model = search.best_estimator_
    else:
        best_model = base_model

else:
    base_model = HistGradientBoostingRegressor(
        max_depth=7,
        learning_rate=0.05,
        max_iter=500,
        l2_regularization=0.0,
        random_state=42,
    )

    base_maes = -cross_val_score(
        base_model,
        X,
        y,
        cv=kf,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    print("HGB base CV MAE mean:", base_maes.mean(), "±", base_maes.std())

    DO_TUNING = True
    if DO_TUNING:
        print("Running RandomizedSearchCV for HistGradientBoostingRegressor...")
        param_dist = {
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.02, 0.03, 0.05, 0.08],
            "max_iter": [300, 500, 800],
            "l2_regularization": [0.0, 0.1, 1.0],
            "min_samples_leaf": [10, 20, 50],
        }

        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_dist,
            n_iter=25,
            cv=3,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            random_state=42,
            verbose=1,
        )
        search.fit(X, y)
        print("Best HGB CV MAE:", -search.best_score_)
        print("Best params:", search.best_params_)
        best_model = search.best_estimator_
    else:
        best_model = base_model


best_model.fit(X_tr, y_tr)
val_pred_gb = best_model.predict(X_val)
gb_mae = mean_absolute_error(y_val, val_pred_gb)
print("Gradient boosting val MAE:", gb_mae)

ensemble_val_pred = 0.5 * val_pred_rf + 0.5 * val_pred_gb
ensemble_mae = mean_absolute_error(y_val, ensemble_val_pred)
print("Ensemble (RF + GB) val MAE:", ensemble_mae)

metrics_path = os.path.join(RESULTS_DIR, "metrics.txt")
with open(metrics_path, "w", encoding="utf-8") as f:
    f.write("PolyML Task 1 – Melting Point Prediction\n")
    f.write("\nValidation metrics (on hold-out split):\n")
    f.write(f"Baseline MAE (predict mean): {baseline_mae:.6f}\n")
    f.write(f"RandomForest val MAE:        {rf_mae:.6f}\n")
    f.write(
        f"RandomForest CV MAE (mean): {rf_cv_maes.mean():.6f} ± {rf_cv_maes.std():.6f}\n"
    )
    f.write(f"Gradient boosting val MAE:   {gb_mae:.6f}\n")
    f.write(f"Ensemble (RF+GB) val MAE:    {ensemble_mae:.6f}\n")
print(f"[INFO] Saved metrics summary to {metrics_path}")

print("Training final gradient boosting model on full data...")
best_model.fit(X, y)


test_pred = best_model.predict(X_test)

submission = pd.DataFrame({"id": test_df["id"], "mpC": test_pred})

submission.to_csv("submission.csv", index=False)
print("Saved submission.csv in project root.")

submission_results_path = os.path.join(RESULTS_DIR, "submission.csv")
submission.to_csv(submission_results_path, index=False)
print(f"[INFO] Saved submission copy to {submission_results_path}")
