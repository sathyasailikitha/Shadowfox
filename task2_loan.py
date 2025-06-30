import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

# Load dataset
data = pd.read_csv('/home/likitha/Downloads/shadowfox/loan_prediction.csv')

# Drop Loan_ID if present
if 'Loan_ID' in data.columns:
    data.drop('Loan_ID', axis=1, inplace=True)

# Encode categorical columns
label_encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Handle missing values
imputer = SimpleImputer(strategy='most_frequent')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Separate features and label
X = data_imputed.drop('Loan_Status', axis=1)

# Train on full dataset (optional, or you can reuse a trained model)
y = data_imputed['Loan_Status']
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X, y)

# Predict for all entries
all_predictions = model.predict(X)
data_imputed['Predicted_Loan_Status'] = ['Approved' if p == 1 else 'Not Approved' for p in all_predictions]

# Save to CSV (optional)
data_imputed.to_csv('loan_predictions_output.csv', index=False)

# Show first few rows
print(data_imputed[['Predicted_Loan_Status']].head())

