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
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import joblib
import os

# ── 1. Load data ──────────────────────────────────────────────
data = pd.read_csv('Data_sets/filtered_data.csv')

# ── 2. Preprocess Years in current job ───────────────────────
data['Years in current job'] = (
    data['Years in current job']
    .astype(str)
    .str.extract(r'(\d+)')
    .astype(float)
)

# ── 3. Use the existing Default column ───────────────────────
data.dropna(subset=['Default'], inplace=True)

# ── 4. Features and target ───────────────────────────────────
features = data.drop(columns=['Id', 'Default'], errors='ignore')
target = data['Default']

# ── 5. One-hot encode categorical columns ────────────────────
categorical_columns = ['Home Ownership', 'Purpose', 'Term']
features = pd.get_dummies(features, columns=categorical_columns)

# ── 6. Save feature columns for app.py to use ────────────────
feature_columns = features.columns.tolist()
joblib.dump(feature_columns, 'ML_MODEL/feature_columns.pkl')
print(f"Saved {len(feature_columns)} feature columns")

# ── 7. Impute missing values ──────────────────────────────────
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)
features_imputed = pd.DataFrame(features_imputed, columns=feature_columns)

# ── 8. Train/test split ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    features_imputed, target, test_size=0.2, random_state=42
)

# ── 9. Define all models as pipelines ────────────────────────
models = {
    'knn_model': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier())
    ]),
    'decision_tree_model': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ]),
    'random_forest_model': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
    'logistic_regression_model': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000))
    ]),
    'naive_bayes_model': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GaussianNB())
    ]),
}

# ── 10. Train and save each model ────────────────────────────
os.makedirs('ML_MODEL', exist_ok=True)

for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipeline.predict(X_test))
    print(f"{name}: Test Accuracy = {acc:.2f}")
    joblib.dump(pipeline, f'ML_MODEL/{name}.pkl')
    print(f"Saved ML_MODEL/{name}.pkl")

print("\nAll models trained and saved successfully!")
