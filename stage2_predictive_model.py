# Climate Change CO2 Emissions - Stage 2: Data Exploration & Predictive Modeling

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 2. Load cleaned dataset
print("Loading cleaned data...")
df = pd.read_csv("clean_climate_data.csv")
print("Loaded:", df.shape)

# 3. Filter for CO2 emissions only
co2_df = df[df['Series name'].str.contains("CO2 emissions")].copy()

# Convert 'Year' to numeric
co2_df['Year'] = pd.to_numeric(co2_df['Year'], errors='coerce')

# 4. Pivot to wide format for modeling
print("Reshaping data...")
co2_pivot = co2_df.pivot_table(index=['Country name', 'Year'],
                               columns='Series name', values='Value').reset_index()

# Drop rows with NaN
co2_pivot.dropna(inplace=True)

# 5. Exploratory Data Analysis - Visualize Trends
plt.figure(figsize=(12, 6))
sns.lineplot(data=co2_pivot, x='Year', y=co2_pivot.columns[2])
plt.title("Global CO2 Emissions Over Time")
plt.xlabel("Year")
plt.ylabel("CO2 Emissions")
plt.tight_layout()
plt.show()

# 6. Add more features for prediction (optional)
# Load full clean dataset again
full_df = pd.read_csv("clean_climate_data.csv")
pivot_df = full_df.pivot_table(index=['Country name', 'Year'],
                                columns='Series name', values='Value').reset_index()

# Drop missing values
pivot_df.dropna(inplace=True)

# 7. Define features and target
target_col = 'CO2 emissions (metric tons per capita)'
if target_col not in pivot_df.columns:
    print(f"Target column '{target_col}' not found.")
else:
    print("Target column found.")

X = pivot_df.drop(columns=['Country name', 'Year', target_col])
y = pivot_df[target_col]

# 8. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. Train Models
# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# 10. Evaluate
print("\nModel Evaluation:")

print("\nLinear Regression")
print("R² Score:", r2_score(y_test, lr_preds))
print("RMSE:", np.sqrt(mean_squared_error(y_test, lr_preds)))

print("\nRandom Forest Regressor")
print("R² Score:", r2_score(y_test, rf_preds))
print("RMSE:", np.sqrt(mean_squared_error(y_test, rf_preds)))

# 11. Predict Future CO2 Emissions (Optional Demo)
print("\nSample Prediction (Random Forest):")
sample = X_test.iloc[0:1]
pred = rf_model.predict(sample)
print("Predicted CO2 Emissions:", pred[0])
