# build_tm_model.py
import os, warnings, time
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

DATA_DIR = "data/"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH  = os.path.join(DATA_DIR, "test.csv")
OUT_SUB = os.path.join(DATA_DIR, "submission_predicted_Tm.csv")
OUT_PLOT = os.path.join(DATA_DIR, "top_feature_scatter.png")
OUT_SHAP = os.path.join(DATA_DIR, "shap_summary.png")

# 1) Load
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

# Auto-detect columns (common names)
smiles_col = [c for c in train.columns if "smiles" in c.lower()]
if len(smiles_col)==0:
    raise ValueError("No SMILES column found in train.csv")
smiles_col = smiles_col[0]
target_col = [c for c in train.columns if c!=smiles_col][0]

print("SMILES column:", smiles_col)
print("Target column:", target_col)
print("Train shape:", train.shape, "Test shape:", test.shape)

# Drop rows with missing SMILES or target
train = train.dropna(subset=[smiles_col, target_col]).reset_index(drop=True)

# 2) Try RDKit
use_rdkit = False
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')  # silence rdkit warnings
    use_rdkit = True
    print("RDKit is available; will compute RDKit descriptors.")
except Exception as e:
    print("RDKit not available, falling back to SMILES heuristics.", e)
    use_rdkit = False

# 3) feature functions
def compute_rdkit_descriptors_safe(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        desc = {
            "MolWt": Descriptors.MolWt(mol),
            "MolLogP": Crippen.MolLogP(mol),
            "TPSA": rdMolDescriptors.CalcTPSA(mol),
            "NumHDonors": rdMolDescriptors.CalcNumHBD(mol),
            "NumHAcceptors": rdMolDescriptors.CalcNumHBA(mol),
            "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "HeavyAtomCount": rdMolDescriptors.CalcNumHeavyAtoms(mol),
            "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
            "NumHeteroatoms": sum(1 for a in mol.GetAtoms() if a.GetSymbol() not in ("C","H")),
            "NumRings": rdMolDescriptors.CalcNumRings(mol)
        }
        return desc
    except Exception:
        return None

def compute_smiles_features(smiles):
    s = str(smiles)
    feat = {"smiles_len": len(s)}
    for atom in ["C","N","O","S","P","F","Cl","Br","I","H"]:
        feat[f"count_{atom}"] = s.count(atom)
    feat["count_equals"] = s.count("=")
    feat["count_hash"] = s.count("#")
    feat["count_aromatic_chars"] = sum(s.count(ch) for ch in "cnosp")
    feat["count_digits"] = sum(ch.isdigit() for ch in s)
    feat["count_slash"] = s.count("/")
    feat["count_backslash"] = s.count("\\")
    return feat

def featurize_list(smiles_list):
    feats = []
    if use_rdkit:
        # RDKit descriptors
        for smi in smiles_list:
            d = compute_rdkit_descriptors_safe(smi)
            if d is None:
                d = {k: np.nan for k in ["MolWt","MolLogP","TPSA","NumHDonors","NumHAcceptors",
                                          "NumRotatableBonds","HeavyAtomCount","NumAromaticRings",
                                          "FractionCSP3","NumHeteroatoms","NumRings"]}
            feats.append(d)
    else:
        for smi in smiles_list:
            feats.append(compute_smiles_features(smi))
    return pd.DataFrame(feats)

# 4) Compute features (train/test)
t0 = time.time()
train_feats = featurize_list(train[smiles_col].astype(str).tolist())
test_feats  = featurize_list(test[smiles_col].astype(str).tolist())
t1 = time.time()
print(f"Featurization done in {t1-t0:.1f}s. train_feats shape: {train_feats.shape}")

# 5) Filter invalid SMILES in train (rows where features are NaN)
valid_mask = ~train_feats.isnull().any(axis=1)
print("Invalid SMILES in train:", (~valid_mask).sum())
train_valid = train[valid_mask].reset_index(drop=True)
train_feats_valid = train_feats[valid_mask].reset_index(drop=True)

# 6) Feature selection (SelectKBest mutual_info)
feature_cols = train_feats_valid.columns.tolist()
X = train_feats_valid[feature_cols].values
y = train_valid[target_col].values
K = min(15, X.shape[1])
selector = SelectKBest(score_func=mutual_info_regression, k=K)
selector.fit(X, y)
selected_idx = selector.get_support(indices=True)
selected_features = [feature_cols[i] for i in selected_idx]
print("Selected features:", selected_features)

# 7) Model pipeline (simple, interpretable-ish)
model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
pipe = Pipeline([("scaler", StandardScaler()), ("selector", selector), ("model", model)])

# 8) CV evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
r2s = cross_val_score(pipe, X, y, scoring="r2", cv=kf, n_jobs=-1)
rmses = np.sqrt(-cross_val_score(pipe, X, y, scoring="neg_mean_squared_error", cv=kf, n_jobs=-1))
maes = -cross_val_score(pipe, X, y, scoring="neg_mean_absolute_error", cv=kf, n_jobs=-1)
print("CV R2 mean:", r2s.mean())
print("CV RMSE mean:", rmses.mean())
print("CV MAE mean:", maes.mean())


# 9) Fit full model
pipe.fit(X, y)

# 10) Permutation importance (faster: n_repeats small)
res = permutation_importance(pipe, X, y, n_repeats=8, random_state=42, n_jobs=-1)
imp_means = res.importances_mean
selected_feature_names = selected_features
perm_imp = dict(zip(selected_feature_names, imp_means))
perm_sorted = sorted(perm_imp.items(), key=lambda x: x[1], reverse=True)
print("Top permutation importances:")
for n,v in perm_sorted[:10]:
    print(n, v)

# 11) Prepare test features: fill NaNs with train medians and predict
test_feats_filled = test_feats.fillna(train_feats_valid.median())
test_X = test_feats_filled[selected_feature_names].values
test_preds = pipe.predict(test_X)

# 12) Save submission
if "id" in test.columns:
    submission = pd.DataFrame({"id": test["id"], "Tm": test_preds})
else:
    submission = pd.DataFrame({smiles_col: test[smiles_col], "Tm": test_preds})
submission.to_csv(OUT_SUB, index=False)
print("Submission saved to:", OUT_SUB)

# 13) Save a quick interpretability plot (top feature vs target)
top_feat = perm_sorted[0][0]
plt.figure(figsize=(6,4))
plt.scatter(train_feats_valid[top_feat], y, s=8)
plt.xlabel(top_feat); plt.ylabel(target_col)
plt.title(f"{target_col} vs {top_feat}")
plt.tight_layout()
plt.savefig(OUT_PLOT)
print("Saved plot to:", OUT_PLOT)

# 14) If SHAP is installed, compute and save SHAP summary (optional)
try:
    import shap
    # get transformed X for the model
    transformed_X = selector.transform(StandardScaler().fit_transform(train_feats_valid[selected_feature_names]))
    # But better to use pipeline parts:
    scaler = pipe.named_steps["scaler"]
    sel = pipe.named_steps["selector"]
    rf = pipe.named_steps["model"]
    X_trans = sel.transform(scaler.transform(train_feats_valid[selected_feature_names]))
    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer.shap_values(X_trans)
    shap.summary_plot(shap_vals, X_trans, feature_names=selected_feature_names, show=False)
    plt.tight_layout()
    plt.savefig(OUT_SHAP)
    print("Saved SHAP to:", OUT_SHAP)
except Exception as e:
    print("SHAP not available or failed:", e)

# Final print summary
print("Done. Summary:")
print("RDKit used:", use_rdkit)
print("Train valid rows:", len(train_valid))
print("Selected features:", selected_features)
print("CV R2 mean:", r2s.mean(),
      "CV RMSE mean:", rmses.mean(),
      "CV MAE mean:", maes.mean())
