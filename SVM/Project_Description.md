# SVC Classification – Iris Dataset (sklearn)

This mini-project trains a **Support Vector Classifier (SVC)** on the classic **Iris** dataset from scikit-learn.

(The educational goal of this project for me was to practice and learn about Support Vectors Classifiers and also practice some hands on Grid-Searching which later was removed from the code before commiting!)

The goal is to classify iris flowers into three species:

- *Iris setosa*
- *Iris versicolor*
- *Iris virginica*

based on four simple measurements of each flower (sepal length/width, petal length/width).

The project follows the same structure as the other mini-projects in this repo: a clean `.py` script, a `results/` folder with plots and metrics, and a focus on understanding the model behavior, not just the accuracy number.

---

## 🔧 What the script does

`iris_svc_classification.py`:

1. **Loads** the Iris dataset via `sklearn.datasets.load_iris(as_frame=True)`
2. **Explores** the data:
   - `iris_pairplot.png` – pairplot of:
     - sepal length (cm)
     - sepal width (cm)
     - petal length (cm)
     - petal width (cm)  
     colored by species (`setosa`, `versicolor`, `virginica`)
   - `iris_corr_heatmap.png` – correlation heatmap of the four numeric features
3. **Prepares features and target**:
   - Features: the 4 numeric measurements (all continuous)
   - Target: `target` (0 = setosa, 1 = versicolor, 2 = virginica)
4. **Splits** the data into train and test sets (stratified by class)
5. **Scales** features with `StandardScaler`
6. **Trains** an `SVC` with an RBF kernel (`C=1.0`, `gamma="scale"`)
7. **Evaluates** the model:
   - Accuracy score
   - Confusion matrix
   - Full classification report (precision, recall, F1-score per class)
   - Saved into `iris_svc_report.txt` and `iris_svc_confusion_matrix.png`
8. **Visualizes decision regions** in 2D:
   - `iris_svc_decision_regions.png` – decision boundaries using petal length vs petal width, with train/test points overlaid

All outputs are saved into the local `results/` directory created next to the script.

---

## 📚 Dataset description (`sklearn.datasets.load_iris`)

This project uses the classic **Iris** dataset, one of the most famous datasets in machine learning.

Key details:

- **Samples**: 150 iris flowers  
- **Classes**: 3 species
  - 0 – *Iris setosa*
  - 1 – *Iris versicolor*
  - 2 – *Iris virginica*
- **Features (all measured in centimeters)**:
  - `sepal length (cm)`
  - `sepal width (cm)`
  - `petal length (cm)`
  - `petal width (cm)`
- **Task**: Multi-class classification – predict the species from these 4 measurements.

The dataset is small, clean, and well-behaved, which makes it ideal for:

- Testing new classification algorithms
- Visualizing decision boundaries
- Understanding how different models separate classes in feature space

In this project, the Iris dataset is loaded directly from scikit-learn (no external CSVs needed) and converted into a pandas DataFrame for easier analysis and plotting.

---

## 📁 Outputs (in `results/`)

After running `iris_svc_classification.py`, the `results/` folder will contain:

**Exploratory plots**

- `iris_pairplot.png`  
  Pairplot of all four features colored by species, useful for seeing which pairs separate classes well (especially petal length vs petal width).

- `iris_corr_heatmap.png`  
  Correlation heatmap showing relationships between the four numeric features.

**Model performance**

- `iris_svc_confusion_matrix.png`  
  Confusion matrix heatmap showing how often each class is correctly or incorrectly predicted.

- `iris_svc_report.txt`  
  Text report with:
  - Overall accuracy
  - Per-class precision, recall, and F1-score
  - Raw confusion matrix values

**Decision boundary**

- `iris_svc_decision_regions.png`  
  2D decision regions using petal length vs petal width, with train and test points, giving an intuitive picture of how SVC separates the three species.

These artifacts make it easy to inspect both the **performance** and the **geometry** of the classifier without running the code yourself.
