# PolyML – Task 1: Melting Point Prediction (mpC)

This project is my solution for **Task 1** of the **PolyML competition**: predicting the **melting point** (in °C, `mpC`) of small organic molecules using their molecular structure encoded as **SMILES** strings.

The goal is to build a **physically meaningful, interpretable model** rather than a pure black box, while still achieving competitive performance.

The final model uses:

- Simple **SMILES string features** (counts of atoms, bonds, rings, etc.)
- Optional **RDKit descriptors** when RDKit is available
- A **Random Forest** baseline
- A tuned **gradient boosting model** (XGBoost or HistGradientBoostingRegressor, depending on availability)
- A simple **ensemble** of RF + gradient boosting on a validation split

All outputs (metrics summary and final submission file) are saved into a `results/` directory for easy inspection and versioning.

---

## 🔧 What the script does

`melting_point_rfr.py`:

1. **Loads the data**  
   - Training data: `data/train.csv`  
   - Test data: `data/test.csv`  

2. **Builds features from SMILES**  
   - **String-based features** (no RDKit required):
     - Length of SMILES
     - Counts of atoms (`C`, `N`, `O`, `S`, halogens, etc.)
     - Aromatic vs non-aromatic counts (`c`, `n`, etc.)
     - Double/triple bond counts (`=`, `#`)
     - Branches and ring indicators (`(`, digits)
     - Simple densities (rings per length, branches per length)
   - **Optional RDKit descriptors** (if RDKit is installed):
     - `MolWt`, `MolLogP`, `TPSA`
     - H-bond donors/acceptors
     - Rotatable bonds
     - Heavy atom count
     - Aromatic rings
     - Fraction sp3 carbons
   - Combines these into a single feature matrix for train and test.
   - Fills missing values with **train medians**.

3. **Baselines & Random Forest sanity check**
   - Baseline MAE by predicting the **mean** of `mpC`.
   - Trains a `RandomForestRegressor` and evaluates:
     - Train/validation split
     - 5-fold cross-validated MAE

4. **Main model: Gradient Boosting**
   - Tries to use **XGBoost** (`XGBRegressor`) with histogram tree method.
   - If XGBoost is not available, falls back to **HistGradientBoostingRegressor** from scikit-learn.
   - Runs **RandomizedSearchCV** over a physically reasonable hyperparameter space:
     - `n_estimators`, `max_depth`, `learning_rate`, subsampling, etc. for XGB
     - `max_depth`, `learning_rate`, `max_iter`, `l2_regularization`, etc. for HGB
   - Selects the **best estimator** based on **MAE**.

5. **Ensemble and validation comparison**
   - Fits the tuned gradient boosting model on the train split (`X_tr, y_tr`).
   - Compares:
     - Random Forest validation MAE
     - Gradient boosting validation MAE
     - A simple **ensemble**: average of RF and GB predictions.
   - Logs these metrics and saves a summary into:
     - `results/metrics.txt`

6. **Final training & submission**
   - Trains the chosen **best gradient boosting model** on **all training data** (`X, y`).
   - Predicts melting points for the test set.
   - Writes a submission file with:
     - `id`
     - `mpC` (predicted melting point)
   - Saves:
     - `submission.csv` (project root – for competition upload)
     - `results/submission.csv` (for the repo and reproducibility)

---

## 🧪 Scientific context: why melting point?

Melting point is a fundamental physical property of matter, critical for numerous scientific and industrial applications, from drug discovery to the creation of advanced materials. However, experimentally measuring this parameter can be a lengthy and expensive process. Can artificial intelligence models help us with this?

### Context

Melting point (\(T_m\)) is a fundamental physical property of matter, critical for numerous scientific and industrial applications, from drug discovery to the creation of advanced materials. Experimental measurement of \(T_m\) can be time-consuming and expensive, and databases, while they exist, are often incomplete.

> Can we predict at what temperature a substance will transition from a solid to a liquid, knowing only the structure of a molecule (SMILES)?

This task uses the **Bradley Double Plus Good Melting Point dataset** and asks us to build a **highly interpretable machine learning model** to predict the melting point of chemical compounds.

---

## 📂 Dataset description

This competition is a **regression** task: predict the melting point of chemical compounds based only on their molecular structure.

The data is derived from the **Jean-Claude Bradley Open Melting Point Dataset**, cleaned and split into train and test:

- `train.csv` – training set with:
  - `smiles` – SMILES string for each molecule
  - `mpC` – target melting point in °C  
- `test.csv` – test set with:
  - `id` – unique identifier
  - `smiles` – SMILES string (no `mpC`, must be predicted)
- `sample_submission.csv` – an example of the expected submission format

### Fields

- `id`  
  Unique identifier of each test example (only present in the test set).

- `smiles`  
  SMILES (Simplified Molecular Input Line Entry System) string representation of the chemical structure.  
  From this string, libraries like **RDKit** can compute physicochemical descriptors (e.g. molecular weight, hydrogen bond counts) or construct a molecular graph.

- `mpC`  
  Target variable: melting point of the compound in degrees Celsius (°C), present only in `train.csv`.

### Data preparation notes

- Records with missing values were removed prior to release.
- The train/test split was performed via a **random split**.
- No additional leakage-specific pre-processing is built into this solution – the script treats the provided CSVs as the ground truth split.

### Useful info for feature engineering

For working with chemical data and converting SMILES into numerical features, it’s standard to use **RDKit**.

Common basic descriptors (also mentioned in the competition statement) include:

- `MolWt` – molecular weight  
- `TPSA` – topological polar surface area  
- `MolLogP` – octanol–water partition coefficient (hydrophilicity vs lipophilicity)  
- `NumHDonors` / `NumHAcceptors` – counts of hydrogen bond donors and acceptors  

This project uses exactly these kinds of descriptors (when RDKit is available), combined with simple SMILES-derived statistics.

---

## ⚛️ Physics & feature engineering hints

Melting point is a **complex** parameter. Unlike boiling point, which correlates reasonably well with molecular mass and surface area, melting point is strongly influenced by how molecules pack in the **crystal lattice**.

Factors that influence crystal stability (and thus \(T_m\)) include:

- **Intermolecular interactions**
  - Hydrogen bonds (H-bond donors/acceptors) strongly increase \(T_m\).
  - Polarity (e.g. dipole moment, TPSA).
- **Rigidity and shape**
  - Rotatable bonds: flexible molecules are harder to pack efficiently below \(T_m\).
  - Aromatic rings and rigid frameworks tend to increase \(T_m\).
- **Symmetry**
  - Highly symmetrical molecules pack better into a crystal, leading to anomalously high melting temperatures (Carnelian effect).
  - Standard fingerprints often miss symmetry, so capturing it explicitly (or via descriptors) is valuable.

The feature set in this project is designed with this physics in mind: counts of heteroatoms, aromatic systems, ring and branching densities, and—when available—RDKit descriptors like TPSA, aromatic ring counts, and rotatable bonds.

---

## 🛡️ Modeling philosophy: interpretability over pure score

In the scientific context, a model isn’t judged only by its accuracy but also by how much **chemical insight** it provides.

The competition emphasizes:

- **Smart feature selection**
  - Prefer a small, physically meaningful set of descriptors (RDKit + handcrafted SMILES statistics) instead of thousands of opaque fingerprint bits.
- **Model simplicity**
  - A **single, well-understood model** is better than a massive ensemble that only improves \(R^2\) at the 4th decimal place.
- **Deep learning guidance**
  - If using neural networks, graph-based models (GNNs) that work directly on molecular topology are preferred over generic dense networks on raw fingerprints.
- **Explainability**
  - Feature importance (e.g. from tree-based models, SHAP values) should be used to understand which chemical patterns drive higher or lower melting points.

This solution follows that spirit by focusing on:

- Chemically interpretable descriptors
- Tree-based models that expose feature importances
- A clear, compact training pipeline

---

## 📁 Outputs (in `results/`)

After running `melting_point_rfr.py`, the `results/` folder contains:

- `metrics.txt`  
  Summary of key validation metrics:
  - Baseline MAE (predicting the mean)
  - Random Forest validation MAE
  - Random Forest cross-validated MAE (mean ± std)
  - Gradient boosting validation MAE
  - Ensemble (RF + GB) validation MAE

- `submission.csv`  
  Final competition submission file with columns:
  - `id`
  - `mpC` (predicted melting point in °C)

This mirrors the competition’s required submission format while keeping everything neatly packaged for Git/GitHub.

---

## 🏆 Competition result & future plans

With this model, I placed **3rd** in this PolyML Task 1 (melting point prediction).  

I’m planning to **tune and refine** the approach further (features + hyperparameters) with the goal of pushing the **mean absolute error down to 20 or below**, while still keeping the model interpretable and chemically meaningful.
