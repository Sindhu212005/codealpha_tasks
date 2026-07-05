import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Create outputs folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)
# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/advertising.csv")
print("First 5 Rows")
print(df.head())
print("\nDataset Information")
print(df.info())
print("\nMissing Values")
print(df.isnull().sum())
print("\nStatistical Summary")
print(df.describe())
# -----------------------------
# Features and Target
# -----------------------------
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]
# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
# -----------------------------
# Train Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)
# -----------------------------
# Prediction
# -----------------------------
y_pred = model.predict(X_test)
# -----------------------------
# Evaluation
# -----------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("\nModel Performance")
print(f"MAE : {mae:.2f}")
print(f"MSE : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R² Score : {r2:.2f}")
# -----------------------------
# Feature Importance
# -----------------------------
importance = pd.DataFrame({"Feature": X.columns,"Coefficient": model.coef_})
print("\nFeature Importance")
print(importance)
# -----------------------------
# Graph 1
# Actual vs Predicted
# -----------------------------
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid(True)
plt.savefig("outputs/output1.png")
plt.show()
plt.close()
# -----------------------------
# Graph 2
# Feature Importance
# -----------------------------
plt.figure(figsize=(6,4))
plt.bar(importance["Feature"], importance["Coefficient"])
plt.xlabel("Features")
plt.ylabel("Coefficient")
plt.title("Feature Importance")
plt.grid(True)
plt.savefig("outputs/output2.png")
plt.show()
plt.close()
# -----------------------------
# Graph 3
# TV vs Sales
# -----------------------------
plt.figure(figsize=(6,4))
plt.scatter(df["TV"], df["Sales"])
plt.xlabel("TV Advertising")
plt.ylabel("Sales")
plt.title("TV vs Sales")
plt.grid(True)
plt.savefig("outputs/output3.png")
plt.show()
plt.close()
# -----------------------------
# Graph 4
# Radio vs Sales
# -----------------------------
plt.figure(figsize=(6,4))
plt.scatter(df["Radio"], df["Sales"])
plt.xlabel("Radio Advertising")
plt.ylabel("Sales")
plt.title("Radio vs Sales")
plt.grid(True)
plt.savefig("outputs/output4.png")
plt.show()
plt.close()
# -----------------------------
# Graph 5
# Newspaper vs Sales
# -----------------------------
plt.figure(figsize=(6,4))
plt.scatter(df["Newspaper"], df["Sales"])
plt.xlabel("Newspaper Advertising")
plt.ylabel("Sales")
plt.title("Newspaper vs Sales")
plt.grid(True)
plt.savefig("outputs/output5.png")
plt.show()
plt.close()
# -----------------------------
# Graph 6
# Sales Distribution
# -----------------------------
plt.figure(figsize=(6,4))
plt.hist(df["Sales"], bins=10)
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.title("Sales Distribution")
plt.grid(True)
plt.savefig("outputs/output6.png")
plt.show()
plt.close()
# -----------------------------
# Predict New Data
# -----------------------------
sample = pd.DataFrame({
    "TV": [150],
    "Radio": [25],
    "Newspaper": [30]
})
prediction = model.predict(sample)
print("\nPredicted Sales for:")
print(sample)
print(f"\nPredicted Sales = {prediction[0]:.2f}")
print("\nProject Completed Successfully!")