# Random Forest – Breast Cancer (sklearn)

This mini project uses a **Random Forest classifier** on the classic **Breast Cancer Wisconsin (Diagnostic)** dataset provided by scikit-learn.

The goal is to classify breast tumors as **malignant** or **benign** based on features computed from digitized images of fine-needle aspirates (FNAs) of breast masses.

It follows the same structure as the other projects in this repo: clean `.py` script, saved plots, and metrics in a `results/` folder so everything can be inspected directly on GitHub.

---

## 🔧 What the script does

`breast_cancer_random_forest.py`:

1. **Loads** the dataset via `sklearn.datasets.load_breast_cancer(as_frame=True)`
2. **Explores** the data by generating:
   - A pairplot of selected features plus the target (`target`)
   - A correlation heatmap for all numeric columns
3. **Prepares features and target**:
   - Features: all 30 numeric attributes describing cell nuclei (e.g. `mean radius`, `mean texture`, `mean smoothness`, etc.)
   - Target: `target` (0 = malignant, 1 = benign)
4. **Splits** the data into train and test sets with stratification
5. **Trains** a `RandomForestClassifier` on the training data
6. **Evaluates** the model on the test data using:
   - Accuracy
   - Confusion matrix
   - Classification report (precision, recall, F1-score)
   - ROC curve and AUC
7. **Analyzes feature importance**:
   - Uses `feature_importances_` from the Random Forest
   - Saves them as plots and tables
8. **Saves results** into a `results/` directory:
   - Plots as `.png`
   - Metrics and importances as `.txt` / `.csv`

---

## 📚 Dataset description (`sklearn.datasets.load_breast_cancer`)

This project uses the **Breast Cancer Wisconsin (Diagnostic)** dataset that comes with scikit-learn.

Key details:

- **Samples**: 569  
- **Classes**:
  - 0 – malignant  
  - 1 – benign  
- **Features**: 30 real-valued features computed from digitized images of FNAs of breast masses.  
  Each sample includes measurements such as:
  - `mean radius`, `mean texture`, `mean perimeter`, `mean area`, `mean smoothness`
  - plus similar statistics for worst and standard error values.
- **Task**: Binary classification – determine whether a tumor is malignant or benign.

The dataset is widely used as a standard benchmark for binary classification algorithms.

---

## 📁 Outputs (in `results/`)

After running the script, you’ll get:

**Exploratory plots**

- `breast_cancer_rf_pairplot.png`  
  Pairplot of selected features (e.g. `mean radius`, `mean texture`, `mean perimeter`, `mean area`, `mean smoothness`) colored by `target`.

- `breast_cancer_rf_corr_heatmap.png`  
  Correlation matrix heatmap showing relationships between all numeric features and the target.

**Model performance**

- `breast_cancer_random_forest_confusion_matrix.png`  
  Confusion matrix heatmap for the Random Forest model.

- `breast_cancer_random_forest_roc_curve.png`  
  ROC curve with AUC value in the legend.

- `breast_cancer_random_forest_report.txt`  
  Accuracy, confusion matrix, and full classification report.

**Feature importance**

- `breast_cancer_random_forest_feature_importances.png`  
  Horizontal bar chart of feature importances from the Random Forest.

- `breast_cancer_random_forest_feature_importances.csv`  
  Table of features and their importance scores (CSV format).

- `breast_cancer_random_forest_feature_importances.txt`  
  Human-readable list of feature importances, one per line.

These artifacts let anyone inspect how well the model performs and which features matter most, without needing to run the code.
