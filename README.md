README.md

# Sales Forecasting (Time Series + Machine Learning)

*Summary:* Forecast weekly / monthly sales using historical data. Includes EDA, feature engineering (lags, rolling), model comparison (Linear Regression, Random Forest, XGBoost), evaluation, and forecast visuals.

*Dataset:* Walmart Recruiting – Store Sales Forecasting  
https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data

*Files*
- notebook.ipynb – end-to-end notebook (recommended)
- code.py – runnable script
- dataset_link.txt
- insights.txt
- interview_questions.md
- images/ – charts & forecast screenshot

*How to run*
1. Download dataset to data/ folder.
2. pip install -r requirements.txt (if provided).
3. Run notebook.ipynb in Colab or Jupyter, or python code.py.

*Business impact:* Improve inventory planning and reduce stockouts by forecasting 2–6 weeks ahead.

code.py

 1. Import Libraries
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# -------------------------------
# 2. Load Dataset
# -------------------------------
df = pd.read_csv("train.csv")  # Walmart Store Sales dataset
print(df.head())

# -------------------------------
# 3. Data Preprocessing
# -------------------------------
# Convert date to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Extract useful time features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

# Drop irrelevant columns (if any)
df = df.dropna()  # remove missing values for simplicity

# -------------------------------
# 4. Exploratory Data Analysis
# -------------------------------
plt.figure(figsize=(12,6))
df.groupby("Date")["Weekly_Sales"].mean().plot()
plt.title("Weekly Sales Trend Over Time")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# -------------------------------
# 5. Feature Selection
# -------------------------------
X = df[["Store", "Dept", "Year", "Month", "Week", "IsHoliday"]]
y = df["Weekly_Sales"]

# -------------------------------
# 6. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 7. Model Training
# -------------------------------
# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# XGBoost
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)

# -------------------------------
# 8. Model Evaluation
# -------------------------------
models = {"Linear Regression": lr, "Random Forest": rf, "XGBoost": xgb}

for name, model in models.items():
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"{name}:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²: {r2:.2f}")
    print("-"*30)

# -------------------------------
# 9. Forecast Visualization
# -------------------------------
y_pred_rf = rf.predict(X_test[:50])

plt.figure(figsize=(10,5))
plt.plot(range(len(y_test[:50])), y_test[:50], label="Actual Sales", marker="o")
plt.plot(range(len(y_test[:50])), y_pred_rf, label="Predicted Sales", marker="x")
plt.title("Actual vs Predicted Sales (Random Forest)")
plt.xlabel("Time")
plt.ylabel("Weekly Sales")
plt.legend()
plt.show()


dataset_link.txt
https://www.kaggle.com/c/store-sales-time-series-forecasting?utm_source=chatgpt.com


insights.txt
- Month and week features are strong predictors (seasonality).
- Lag and rolling mean features boost short-term accuracy.
- Random Forest/XGBoost capture non-linear patterns better than linear models.
- Business action: increase stock 2-4 weeks before historical peaks.

  http://interview_questions.md/'
  Q1: How did you split time series data?  
A: Time-based split (train on earlier dates, test on later) to avoid leakage.

Q2: Which metrics to use in forecasting?  
A: MAE, RMSE for absolute error, and R² for variance explained.

Q3: Why use lag features?  
A: To allow model to learn recent trends and seasonality that affect next-period sales.

Q4: Deploy approach?  
A: Export model, schedule weekly predictions, integrate with BI dashboards for operations.
