# KNN Classification – Practice Project

## Situation and data description:
You've been given a classified data set from a company! They've hidden the feature column names but have given you the data and the target classes.
We'll try to use KNN to create a model that directly predicts a class for a new data point based off of the features.

This project is a small end-to-end example of using **k-Nearest Neighbors (KNN)** for binary classification in Python.  
It focuses on data preprocessing, model training, hyperparameter tuning, and basic result visualization.

## 🔧 What the script does

- Loads the dataset from `KNN_Project_Data`
- Scales features using `StandardScaler`
- Splits data into train/test sets
- Trains a baseline KNN model (k = 5)
- Tunes K over a range of values and plots **error rate vs k**
- Trains a final KNN model (k = 30 by default)
- Saves:
  - Plots to `results/` (`knn_pairplot.png`, `knn_error_vs_k.png`)
  - Evaluation reports (confusion matrix, classification report, accuracy) as `.txt` files

## ▶️ How to run

From this folder:
After running, check the results/ directory for:
Visualizations of the data and model performance
Text files containing detailed evaluation metrics

