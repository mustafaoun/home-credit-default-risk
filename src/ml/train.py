import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import joblib

# Load data
df = pd.read_csv('../application_train.csv')

# Drop high missing columns
missing_pct = df.isnull().mean() * 100
cols_to_drop = missing_pct[missing_pct > 50].index.tolist()
df_clean = df.drop(columns=cols_to_drop)

# Handle DAYS_EMPLOYED
df_clean['DAYS_EMPLOYED_MISSING'] = df_clean['DAYS_EMPLOYED'].isnull().astype(int)
df_clean['DAYS_EMPLOYED'] = df_clean['DAYS_EMPLOYED'].fillna(0)

# Debt to income
df_clean['DEBT_TO_INCOME'] = df_clean['AMT_CREDIT'] / df_clean['AMT_INCOME_TOTAL']

# Features and target
X = df_clean.drop(columns=['TARGET', 'SK_ID_CURR'])
y = df_clean['TARGET']

# Columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = [c for c in X.select_dtypes(include=['object']).columns if X[c].nunique() < 10]

# Preprocessor
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ]), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Pipeline
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'))
])

# Train
final_pipeline.fit(X_train, y_train)

# Score
y_pred_proba = final_pipeline.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC: {auc:.4f}")

# Save
joblib.dump(final_pipeline, '../model.pkl')
print("Model saved as model.pkl")

