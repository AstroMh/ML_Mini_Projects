# ================================
# PolyML: mpC prediction script (improved)
# ================================

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error

# --------------------------------
# 1) Load data
# --------------------------------
train_df = pd.read_csv('data/train.csv')
test_df  = pd.read_csv('data/test.csv')

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print(train_df.head())

# --------------------------------
# 2) Feature engineering
# --------------------------------

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
        feat["num_N"] + feat["num_O"] + feat["num_S"] +
        feat["num_F"] + feat["num_Cl"] + feat["num_Br"] + feat["num_I"]
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
except Exception as e:
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


print("Building features...")
train_features = build_features(train_df)
test_features  = build_features(test_df)
print("Train features shape:", train_features.shape)
print("Test features shape:", test_features.shape)

# Align columns (just in case)
common_cols = train_features.columns.intersection(test_features.columns)
train_features = train_features[common_cols]
test_features  = test_features[common_cols]

# Fill any NaNs with train medians
median_vals = train_features.median()
train_features = train_features.fillna(median_vals)
test_features  = test_features.fillna(median_vals)

X = train_features.values
y = train_df["mpC"].values
X_test = test_features.values

# --------------------------------
# 3) Quick baseline & sanity check (RandomForest)
# --------------------------------
from sklearn.ensemble import RandomForestRegressor

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
    n_jobs=-1
)
rf_baseline.fit(X_tr, y_tr)
val_pred_rf = rf_baseline.predict(X_val)
rf_mae = mean_absolute_error(y_val, val_pred_rf)
print("Baseline RandomForest MAE:", rf_mae)

# --------------------------------
# 4) Gradient boosting model (XGBoost if available, else HistGradientBoosting)
# --------------------------------
use_xgb = False
try:
    from xgboost import XGBRegressor
    use_xgb = True
    print("Using XGBoostRegressor as main model.")
except Exception as e:
    print("xgboost not available, using HistGradientBoostingRegressor instead.")
    from sklearn.ensemble import HistGradientBoostingRegressor

# Cross-validation splitter
kf = KFold(n_splits=5, shuffle=True, random_state=42)

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
        n_jobs=-1
    )

    base_maes = -cross_val_score(
        base_model,
        X, y,
        cv=kf,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
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
    # HistGradientBoostingRegressor path
    base_model = HistGradientBoostingRegressor(
        max_depth=7,
        learning_rate=0.05,
        max_iter=500,
        l2_regularization=0.0,
        random_state=42
    )

    base_maes = -cross_val_score(
        base_model,
        X, y,
        cv=kf,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
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

# --------------------------------
# 5) Final train/val check on best_model
# --------------------------------
best_model.fit(X_tr, y_tr)
val_pred_gb = best_model.predict(X_val)
gb_mae = mean_absolute_error(y_val, val_pred_gb)
print("Gradient boosting val MAE:", gb_mae)

# Better ensemble: average RF + GB on validation (just to inspect)
ensemble_val_pred = 0.5 * val_pred_rf + 0.5 * val_pred_gb
ensemble_mae = mean_absolute_error(y_val, ensemble_val_pred)
print("Ensemble (RF+GB) val MAE:", ensemble_mae)

# --------------------------------
# 6) Train final model on full data (choose GB or ensemble)
# --------------------------------
# For submission, use the stronger model: GB or ensemble.
# Kaggle submission must be a single prediction, so we'll use GB (often stronger).

print("Training final gradient boosting model on full data...")
best_model.fit(X, y)

# --------------------------------
# 7) Predict test and save submission
# --------------------------------
test_pred = best_model.predict(X_test)

submission = pd.DataFrame({
    "id": test_df["id"],
    "mpC": test_pred
})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")
