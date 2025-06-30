import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('/home/likitha/Downloads/shadowfox/HousingData.csv')

# Clean column names (strip whitespace)
df.columns = df.columns.str.strip()

# Show dataset info
print("First 5 rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

# Try to detect 'MEDV' or alternative target column
possible_targets = ['MEDV', 'Price', 'Target', 'HousePrice']
target_column = None

for col in df.columns:
    if col.upper() in [t.upper() for t in possible_targets]:
        target_column = col
        break

if not target_column:
    raise ValueError("Could not find a target column like 'MEDV' or 'Price'. Please check your dataset.")

print(f"\nUsing target column: {target_column}")

# Split features and target
X = df.drop(target_column, axis=1)
y = df[target_column]

# Check for non-numeric columns
non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric_cols:
    print(f"Warning: Dropping non-numeric columns: {non_numeric_cols}")
    X = X.drop(columns=non_numeric_cols)

# Handle missing values
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# Ensure all inputs are numeric
X = X.astype(float)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Plot actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.show()

