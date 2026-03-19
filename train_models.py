import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
import os

# ── 1. Load data ──────────────────────────────────────────────
data = pd.read_csv('Data_sets/filtered_data.csv')
print(f"Dataset shape: {data.shape}")

# ── 2. Preprocess Years in current job ───────────────────────
data['Years in current job'] = (
    data['Years in current job']
    .astype(str)
    .str.extract(r'(\d+)')
    .astype(float)
)

# ── 3. Better target variable ─────────────────────────────────
def calculate_default(row):
    return 1 if (
        row['Credit Score'] < 650 or
        row['Number of Credit Problems'] > 1 or
        row['Tax Liens'] > 0 or
        row['Bankruptcies'] > 0 or
        (row['Monthly Debt'] > 0 and row['Annual Income'] > 0 and
         (row['Monthly Debt'] / (row['Annual Income'] / 12)) > 0.5)
    ) else 0

data['Default'] = data.apply(calculate_default, axis=1)
print(f"Default distribution:\n{data['Default'].value_counts()}")

# ── 4. Keep only important features ─────────────────────────
features = data[[
    'Credit Score',
    'Annual Income', 
    'Monthly Debt',
    'Tax Liens',
    'Bankruptcies',
    'Number of Credit Problems',
    'Home Ownership',
    'Purpose',
    'Term'
]]
target = data['Default']

# ── 5. One-hot encode categorical columns ────────────────────
categorical_columns = ['Home Ownership', 'Purpose', 'Term']
features = pd.get_dummies(features, columns=categorical_columns)

# ── 6. Save feature columns ───────────────────────────────────
feature_columns = features.columns.tolist()
joblib.dump(feature_columns, 'ML_MODEL/feature_columns.pkl')
print(f"Saved {len(feature_columns)} feature columns")

# ── 7. Impute missing values ──────────────────────────────────
imputer = SimpleImputer(strategy='median')
features_imputed = imputer.fit_transform(features)
features_imputed = pd.DataFrame(features_imputed, columns=feature_columns)
joblib.dump(imputer, 'ML_MODEL/imputer.pkl')

# ── 8. Train/test split ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    features_imputed, target, test_size=0.2, random_state=42, stratify=target
)

# ── 9. Handle class imbalance with SMOTE ─────────────────────
print(f"\nBefore SMOTE: {y_train.value_counts().to_dict()}")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {pd.Series(y_train_bal).value_counts().to_dict()}")

# ── 10. Define all models as pipelines ───────────────────────
models = {
    'knn_model': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier(n_neighbors=5))
    ]),
    'decision_tree_model': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', DecisionTreeClassifier(
            random_state=42,
            max_depth=10,          # prevent overfitting
            min_samples_split=10,
            min_samples_leaf=5
        ))
    ]),
    'random_forest_model': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,          # prevent overfitting
            min_samples_split=10,
            min_samples_leaf=5
        ))
    ]),
    'logistic_regression_model': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, C=0.1))
    ]),
    'naive_bayes_model': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        ))
    ]),
}

# ── 11. Train and save each model ────────────────────────────
os.makedirs('ML_MODEL', exist_ok=True)

print("\n--- Model Results ---")
for name, pipeline in models.items():
    pipeline.fit(X_train_bal, y_train_bal)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name}: Test Accuracy = {acc:.2f}")
    print(classification_report(y_test, y_pred))
    joblib.dump(pipeline, f'ML_MODEL/{name}.pkl')
    print(f"Saved ML_MODEL/{name}.pkl")

print("\nAll models trained and saved successfully!")