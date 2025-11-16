# Linear Regression – Ecommerce Customers

This project uses **Linear Regression** to predict the **Yearly Amount Spent** by ecommerce customers based on their behavior and account data.

It’s a clean, end-to-end example of:
- Exploratory data analysis
- Feature selection
- Training a linear regression model
- Evaluating regression performance
- Saving results (plots + metrics) for viewing on GitHub

---

## 🔧 What the script does

`linear_regression_project.py`:

1. **Loads** the dataset from `Ecommerce Customers`
2. **Cleans** it by dropping non-numeric text columns:
   - `Address`, `Email`, `Avatar`
3. **Explores** the data by generating and saving:
   - A pairplot of all numeric features
   - A correlation heatmap
4. **Prepares features**:
   - `Avg. Session Length`
   - `Time on App`
   - `Time on Website`
   - `Length of Membership`
5. **Trains** a `LinearRegression` model on train data
6. **Evaluates** the model on test data using:
   - MAE, MSE, RMSE
7. **Generates plots**:
   - Predicted vs Actual yearly spending
   - Residuals distribution
8. **Saves coefficients** for each feature

All results are saved in the local `results/` folder.

---

## 📁 Outputs (in `results/`)

After running the script, you’ll get:

**Plots**
- `linear_regression_pairplot.png` – pairplot of numeric features  
- `linear_regression_corr_heatmap.png` – correlation matrix heatmap  
- `linear_regression_pred_vs_actual.png` – predicted vs actual yearly amount spent  
- `linear_regression_residuals.png` – residuals (actual − predicted) distribution  

**Text & Tables**
- `linear_regression_report.txt` – MAE, MSE, RMSE  
- `linear_regression_coefficients.csv` – feature names + coefficients (CSV)  
- `linear_regression_coefficients.txt` – human-readable list of coefficients  
