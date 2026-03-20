# Exhibit Art Shipping Cost Prediction

This project is designed to predict the total cost required to ship sculptures to customers. It was developed as a solution for a HackerEarth machine learning challenge, focusing on optimizing logistics costs for a transport company.

## 📌 Problem Statement
The goal is to analyze historical shipment data—including artist reputation, sculpture physical attributes, and shipping logistics—to build a predictive model for future shipping costs. Accurate predictions help the company:
* Prioritize shipments for high-reputation artists.
* Optimize batch transport for nearby locations.
* Avoid losses by ensuring customer quotes are profitable and accurate.

## 📂 Project Structure
* `Hacker earth exhibit art.py`: The main Python script containing data cleaning, statistical testing, and model training.
* `train (1).csv`: Training dataset with ~6,500 records.
* `test.csv`: Test dataset with ~3,500 records for final predictions.
* `sample_submission.csv`: The final output containing predicted costs for the test set.

## 🛠️ Technical Workflow

### 1. Exploratory Data Analysis (EDA)
The script performs extensive visualization to understand data distribution and outliers:
* **Correlation Matrix**: Identifying strong relationships (e.g., Height vs. Width, Weight vs. Price).
* **Visualizations**: Uses `Seaborn` and `Matplotlib` for Box plots, Violin plots, and Swarm plots to detect skewness and outliers.
* **Dendrograms**: Hierarchical clustering to visualize feature groupings.

### 2. Statistical Hypothesis Testing
Unlike standard ML pipelines, this project implements formal statistical tests for feature selection:
* **Chi-Square Test**: Used to determine dependencies between categorical variables (e.g., `Material` and `Fragile`).
* **VIF (Variance Inflation Factor)**: Detects multicollinearity in numerical features. Features with high VIF (like raw dimensions) were transformed or flagged.
* **ANOVA Test**: Checks if different categories (like `Transport` type) have a statistically significant impact on the mean shipping `Cost`.

### 3. Preprocessing & Feature Engineering
* **Missing Value Imputation**: Uses mean/median for numerical data and mode for categorical data.
* **Log Transformation**: Applied to `Weight`, `Height`, and `Price` to normalize skewed distributions.
* **Date Engineering**: Calculates the difference between `Scheduled Date` and `Delivery Date`.
* **Encoding**: Implements One-Hot Encoding via the `feature_engine` library.

### 4. Machine Learning Models
Multiple algorithms were trained and compared:
* Linear Regression
* K-Neighbors Regressor
* Gradient Boosting Regressor
* **Random Forest Regressor** (Final Model chosen for its 96% test accuracy)
* XGBoost & LightGBM

## 🚀 How to Run
1.  **Install Dependencies**:
    ```bash
    pip install pandas numpy seaborn matplotlib sklearn statsmodels feature_engine xgboost lightgbm
    ```
2.  **Execute the Script**:
    ```bash
    python "Hacker earth exhibit art.py"
    ```

## 📊 Results
The final **Random Forest Regressor** achieved:
* **Training Accuracy**: ~99%
* **Test Accuracy**: ~96%
* The predictions were exported using an exponential transformation (`np.exp`) to revert the log-scaling applied during training.

---

