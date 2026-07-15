# task2_pipeline/churn_pipeline.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

print("🚀 Step 1: Downloading Telco Churn Dataset...")
url = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(url)

print("🧹 Step 2: Preprocessing raw features...")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(" ", np.nan))
df = df.drop(columns=['customerID'])
df['Churn'] = df['Churn'].astype(str).str.lower().map({'yes': 1, 'no': 0}).fillna(0)

X = df.drop(columns=['Churn'])
y = df['Churn']

numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [col for col in X.columns if col not in numeric_features]
X[categorical_features] = X[categorical_features].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("⚙️ Step 3: Setting up the Sklearn ColumnTransformer...")
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

print("🏗️ Step 4: Building model training Pipeline...")
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [5, 10]
}

print("⚡ Step 5: Executing Hyperparameter Tuning with GridSearchCV...")
grid_search = GridSearchCV(full_pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"✅ GridSearch Complete! Best parameters: {grid_search.best_params_}")

print("\n📊 Step 6: Pipeline Evaluation Metrics:")
predictions = best_model.predict(X_test)
print(classification_report(y_test, predictions))

print("💾 Step 7: Exporting complete pipeline binary...")
pipeline_filename = 'telco_churn_pipeline.joblib'
joblib.dump(best_model, pipeline_filename)
print(f"🎉 Pipeline successfully saved to '{pipeline_filename}'!")