# Logistic Regression – Advertising Click Prediction

In this project we work with a **fake advertising dataset**, indicating whether or not a particular internet user clicked on an advertisement.  
The goal is to build a **logistic regression model** that predicts whether a user will click on an ad based on their characteristics and behavior.

This mini-project demonstrates:

- Binary classification with logistic regression
- Basic exploratory data analysis (EDA)
- Feature selection and preprocessing
- Model evaluation with accuracy, confusion matrix, classification report, and ROC curve
- Saving results to files so they can be viewed directly on GitHub

---

## 🔧 What the script does

`advertising_logistic_regression.py`:

1. **Loads** the dataset from `advertising.csv`
2. **Explores** the data by generating:
   - A pairplot of key numeric features plus the target (`Clicked on Ad`)
   - A correlation heatmap of numeric features
3. **Prepares features and target**:

   **Features used (numeric):**
   - `Daily Time Spent on Site` – minutes per day on the website  
   - `Age` – user’s age in years  
   - `Area Income` – average income of the user’s geographical area  
   - `Daily Internet Usage` – minutes per day on the internet  
   - `Male` – binary flag (1 = male, 0 = female)

   **Target:**
   - `Clicked on Ad` – binary label (0 = did not click, 1 = clicked)

   Text fields such as `Ad Topic Line`, `City`, `Country`, and `Timestamp` are **not** used in this basic model.

4. **Splits** the data into train and test sets (stratified by `Clicked on Ad`)
5. **Scales** the numeric features using `StandardScaler`
6. **Trains** a `LogisticRegression` model
7. **Evaluates** the model using:
   - Accuracy
   - Confusion matrix
   - Classification report (precision, recall, F1-score)
   - ROC curve and AUC
8. **Saves** plots and reports into a `results/` directory:
   - Confusion matrix heatmap
   - ROC curve
   - Metrics and coefficients as `.txt` / `.csv`

---

## 📚 Dataset description

This is a **synthetic (fake) advertising dataset** commonly used for practice.  
Each row represents a single internet user with the following typical columns:

- `Daily Time Spent on Site` – time spent on the website (minutes/day)  
- `Age` – user’s age  
- `Area Income` – average income in the user’s area  
- `Daily Internet Usage` – total internet usage (minutes/day)  
- `Ad Topic Line` – short text description of the ad  
- `City` – user’s city  
- `Male` – indicator (1 = male, 0 = female)  
- `Country` – user’s country  
- `Timestamp` – date and time of the page visit  
- `Clicked on Ad` – target label, whether the user clicked the ad (0/1)

The main task is to **predict `Clicked on Ad`** from the user’s demographic and behavioral features.

---

## 📁 Outputs (in `results/`)

After running the script, you’ll get:

**Plots**

- `advertising_pairplot.png`  
  Pairplot of numeric features (Daily Time Spent on Site, Age, Area Income, Daily Internet Usage, Male) colored by `Clicked on Ad`.

- `advertising_corr_heatmap.png`  
  Correlation heatmap showing relationships between numeric features and the target.

- `advertising_confusion_matrix.png`  
  Confusion matrix heatmap for the logistic regression model.

- `advertising_roc_curve.png`  
  ROC curve with the AUC value displayed in the legend.

**Text & Tables**

- `advertising_logistic_report.txt`  
  Accuracy, confusion matrix, and full classification report.

- `advertising_logistic_coefficients.csv`  
  Table of feature names and their corresponding logistic regression coefficients.

- `advertising_logistic_coefficients.txt`  
  Human-readable list of feature coefficients, one per line.

These files can be viewed directly on GitHub so anyone can inspect the model’s performance and interpretation without running the code.
