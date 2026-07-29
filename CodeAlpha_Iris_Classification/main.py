import pandas as pd
import matplotlib.pyplot as plt
# Load the dataset
df = pd.read_csv("data/Iris.csv")
# Display first 5 rows
print("First 5 Rows of the Dataset:")
print(df.head())
# Dataset information
print("\nDataset Information:")
print(df.info())
# Remove Id column if it exists
if "Id" in df.columns:
    df = df.drop("Id", axis=1)
# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())
# Statistical summary
print("\nStatistical Summary:")
print(df.describe())
# Plot histograms
df.hist(figsize=(10, 8))
plt.suptitle("Iris Dataset Histograms")
plt.tight_layout()
plt.show()
# Import Machine Learning libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
# Features and target
X = df.drop("Species", axis=1)
y = df["Species"]
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.3,random_state=42)
# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
plt.show()
input("\nPress Enter to exit...")