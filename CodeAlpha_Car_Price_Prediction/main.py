import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Load Dataset
df = pd.read_csv("data/car_data.csv")
# Display Dataset
print("First 5 Rows:")
print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())
# Feature Engineering
current_year = 2025
df["Car_Age"] = current_year - df["Year"]
# Drop unnecessary columns
df.drop(["Car_Name", "Year"], axis=1, inplace=True)
# Convert categorical columns into numerical values
df = pd.get_dummies(df, drop_first=True)
print("\nProcessed Dataset:")
print(df.head())
# Features and Target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]
# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
# ---------------- Linear Regression ----------------
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print("\n========== Linear Regression ==========")
print("MAE :", mean_absolute_error(y_test, lr_pred))
print("MSE :", mean_squared_error(y_test, lr_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, lr_pred)))
print("R2 Score:", r2_score(y_test, lr_pred))
# ---------------- Random Forest ----------------
rf = RandomForestRegressor(n_estimators=100,random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("\n========== Random Forest ==========")
print("MAE :", mean_absolute_error(y_test, rf_pred))
print("MSE :", mean_squared_error(y_test, rf_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, rf_pred)))
print("R2 Score:", r2_score(y_test, rf_pred))
# ---------------- Graph 1 ----------------
plt.figure(figsize=(8,6))
plt.scatter(y_test, rf_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.grid(True)
plt.show()
# ---------------- Graph 2 ----------------
importance = pd.Series(rf.feature_importances_, index=X.columns)
importance.sort_values().plot(kind="barh", figsize=(10,6))
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.show()
# ---------------- Sample Predictions ----------------
results = pd.DataFrame({"Actual Price": y_test.values,
    "Predicted Price": rf_pred})
print("\nSample Predictions:")
print(results.head(10))