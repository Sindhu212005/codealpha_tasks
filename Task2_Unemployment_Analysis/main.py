import pandas as pd
import matplotlib.pyplot as plt
# Load dataset
df = pd.read_csv("data/unemployment.csv")
# Display first 5 rows
print("First 5 Rows:")
print(df.head())
# Dataset information
print("\nDataset Info:")
print(df.info())
# Missing values
print("\nMissing Values:")
print(df.isnull().sum())
# Convert Date column to datetime
df[" Date"] = pd.to_datetime(df[" Date"], dayfirst=True)
# Remove extra spaces from column names
df.columns = df.columns.str.strip()
# Average unemployment rate by state
state_avg = df.groupby("Region")["Estimated Unemployment Rate (%)"].mean().sort_values(ascending=False)
print("\nAverage Unemployment Rate by State:")
print(state_avg)
# Graph 1: Top 10 States
plt.figure(figsize=(10,6))
state_avg.head(10).plot(kind="bar")
plt.title("Top 10 States by Average Unemployment Rate")
plt.xlabel("State")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("top10_states.png")
plt.show()
# Graph 2: Monthly Trend
monthly = df.groupby("Date")["Estimated Unemployment Rate (%)"].mean()
plt.figure(figsize=(10,6))
plt.plot(monthly.index, monthly.values, marker="o")
plt.title("Monthly Average Unemployment Rate")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("monthly_trend.png")
plt.show()
# Graph 3: Rural vs Urban
area_avg = df.groupby("Area")["Estimated Unemployment Rate (%)"].mean()
plt.figure(figsize=(6,6))
area_avg.plot(kind="bar")
plt.title("Average Unemployment Rate by Area")
plt.xlabel("Area")
plt.ylabel("Unemployment Rate (%)")
plt.tight_layout()
plt.savefig("area_comparison.png")
plt.show()
# Highest unemployment state
highest = state_avg.idxmax()
print("\nState with Highest Average Unemployment:", highest)
# Lowest unemployment state
lowest = state_avg.idxmin()
print("State with Lowest Average Unemployment:", lowest)
print("\nProject Completed Successfully!")