# Machine Learning Practice Projects (Python)

This repository collects my personal **machine learning** and **computer vision** practice projects in Python.  
The idea is simple: take real datasets, build clear baseline models, experiment a bit, and keep everything in clean, reproducible **`.py` scripts** instead of messy half-finished notebooks.

---

## 🔍 What’s inside

- **Self-contained mini-projects**  
  Each project lives in its own folder and usually includes:
  - A main Python script (converted from Jupyter notebooks)
  - A local `results/` directory with plots & evaluation metrics
  - A small dataset (CSV or sklearn toy dataset), or a link to the source
  - A project-specific `README.md` explaining what was done

- **Topics covered (so far)**  
  The repo is still growing, but already includes projects on:
  - **k-Nearest Neighbors (KNN)** – tabular data & digits (mini-MNIST)
  - **Linear regression** – house prices, ecommerce spending, etc.
  - **Logistic regression** – classification tasks like ad clicks
  - **Random forests** – tabular classification & regression
  - **Model evaluation** – train/test splits, MAE, accuracy, ROC curves, confusion matrices
  - **Data preprocessing & feature engineering** – scaling, feature selection, handcrafted features
  - **Data visualization** – histograms, pairplots, correlation heatmaps, etc.

More advanced topics (ensembles, cross-validation, interpretability, and some CV/vision tasks) are being added over time.

---

## 🗂 Project structure

Most projects follow this pattern:

- `<project_name>/`
  - `main_script.py` – the main training / evaluation pipeline
  - `results/` – saved plots, metrics, and sometimes model outputs
  - `README.md` – short description of the problem, approach, and how to read the results
  - (optionally) small data files or instructions on how to download them

You can open any folder, read the local README, and run the script directly to reproduce the results.

---

## 📚 Why this repo exists

This repo is mainly a **learning log**:

- To practice core ML concepts on real datasets  
- To keep code organized and reproducible  
- To build a portfolio of small, focused projects that show **how** models are built, not just final scores

Over time, the plan is to grow this into a solid collection of baseline solutions for common ML tasks, plus a few competition-style projects with more detailed feature engineering and model tuning.
