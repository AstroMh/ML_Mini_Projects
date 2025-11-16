# KNN Classification – Digits (sklearn `load_digits`)

This project uses **k-Nearest Neighbors (KNN)** to classify images from the classic `load_digits` dataset in scikit-learn (a small, MNIST-like handwritten digits dataset).

It’s an end-to-end example of:

- Loading a built-in dataset from `sklearn`
- Training and tuning a KNN classifier
- Evaluating performance with accuracy and a confusion matrix
- Visualizing digits, predictions, and error vs. K
- Saving all results to files so they can be viewed directly on GitHub

---

## 🔧 What the script does

`knn_digits_project.py`:

1. **Loads** the digits dataset via `sklearn.datasets.load_digits`
2. **Visualizes** example digits in a grid
3. **Splits** the data into train/test sets (stratified by label)
4. **Trains** a baseline KNN model (default `k = 5`)
5. **Tunes** K over a range (e.g. `k = 1..19`) and plots **error rate vs K**
6. **Trains a final model** using the best K from tuning
7. **Evaluates** the model:
   - Accuracy
   - Confusion matrix
   - Full classification report
8. **Saves**:
   - Plots (sample digits, error vs K, confusion matrix, sample predictions)
   - Text reports with metrics

All results are saved inside a local `results/` directory.

---

## 📚 Dataset description (`sklearn.datasets.load_digits`)

This project uses the **Digits** dataset that comes with scikit-learn:

- **Samples**: 1,797 images  
- **Classes**: digits `0` through `9`  
- **Image size**: `8 × 8` grayscale pixels  
- **Features**: each image is flattened into a 64-dimensional vector  
- **Pixel values**: integers from 0 to 16 representing intensity  
- **Task**: multi-class classification – predict which digit (0–9) is written  

The dataset is based on the **“Optical Recognition of Handwritten Digits”** data from the UCI Machine Learning Repository, preprocessed into 8×8 images for convenient experimentation.

---

## 📁 Outputs (in `results/`)

After running the script, you’ll get:

**Plots**

- `digits_sample_grid.png`  
  Grid of example handwritten digit images with their true labels.

- `digits_error_vs_k.png`  
  Error rate as a function of K (helps visualize the best K).

- `digits_confusion_matrix_k5.png`  
  Confusion matrix heatmap for the **baseline** KNN model (k = 5).

- `digits_confusion_matrix_k<best_k>.png`  
  Confusion matrix heatmap for the **final** KNN model using the best K.

- `digits_sample_predictions_k<best_k>.png`  
  Grid of test images with **predicted vs true** labels; misclassifications highlighted.

**Text reports**

- `baseline_knn_digits_k5.txt`  
  Accuracy, confusion matrix, and classification report for the baseline model (k = 5).

- `final_knn_digits_k<best_k>.txt`  
  Accuracy, confusion matrix, and classification report for the final tuned model.

These files can be viewed directly on GitHub so anyone can inspect the performance and diagnostics without running the code.
