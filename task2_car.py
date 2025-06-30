import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Step 1: Load the dataset
df = pd.read_csv("/home/likitha/Downloads/shadowfox/car.csv")  

# Step 2: Feature engineering
df['car_age'] = 2025 - df['Year']
df.drop(['Year', 'Car_Name'], axis=1, inplace=True)

# Step 3: Define features and target
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Step 4: Preprocessing + model pipeline
cat_features = ['Fuel_Type', 'Seller_Type', 'Transmission']
preprocessor = ColumnTransformer([
    ('onehot', OneHotEncoder(drop='first'), cat_features)
], remainder='passthrough')

model = Pipeline([
    ('preprocessing', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Step 5: Train-test split and model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Step 6: Predict selling price for all cars
predictions = model.predict(X)

# Step 7: Append predictions to original dataset
df_with_predictions = df.copy()
df_with_predictions['Predicted_Price'] = predictions

# Step 8: Save to CSV
df_with_predictions.to_csv("car_price_predictions.csv", index=False)

# Optional: Show first 5 rows
print(df_with_predictions[['Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type',
                           'Transmission', 'Owner', 'car_age', 'Selling_Price', 'Predicted_Price']].head())

