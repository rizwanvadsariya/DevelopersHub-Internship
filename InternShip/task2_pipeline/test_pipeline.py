# task2_pipeline/test_pipeline.py
import pandas as pd
import joblib

print("🔄 Loading your exported Churn Pipeline...")
# Load the saved pipeline
pipeline = joblib.load('telco_churn_pipeline.joblib')

# Raw, dirty data representing a brand new customer (unprocessed)
new_customer_raw = {
    'gender': ['Female'],
    'SeniorCitizen': [0],
    'Partner': ['No'],
    'Dependents': ['No'],
    'tenure': [2],  # Very new customer (risky!)
    'PhoneService': ['Yes'],
    'MultipleLines': ['No'],
    'InternetService': ['Fiber optic'],
    'OnlineSecurity': ['No'],
    'OnlineBackup': ['No'],
    'DeviceProtection': ['No'],
    'TechSupport': ['No'],
    'StreamingTV': ['Yes'],
    'StreamingMovies': ['Yes'],
    'Contract': ['Month-to-month'],  # No commitment (risky!)
    'PaperlessBilling': ['Yes'],
    'PaymentMethod': ['Electronic check'],
    'MonthlyCharges': [90.5],
    'TotalCharges': [181.0]
}

# Convert to DataFrame
df_new = pd.DataFrame(new_customer_raw)

print("🔮 Passing raw data straight into the pipeline...")
# The pipeline automatically handles imputation, scaling, encoding, and predicting!
prediction = pipeline.predict(df_new)[0]
probability = pipeline.predict_proba(df_new)[0][1]

print("\n=================== TEST RESULT ===================")
if prediction == 1:
    print("⚠️  Warning: Customer is highly likely to CHURN!")
else:
    print("✅  Success: Customer is likely to STAY (No Churn).")
print(f"Confidence Rate: {probability:.2%}")
print("===================================================")