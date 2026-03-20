import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# ── 1. Load data ──────────────────────────────────────────────
data = pd.read_csv('Data_sets/filtered_data.csv')
print(f"Dataset shape: {data.shape}")

# ── 2. Drop ID column ─────────────────────────────────────────
data.drop(columns=['Id'], inplace=True, errors='ignore')

# ── 3. Clean Credit Score (remove outliers above 850) ─────────
data['Credit Score'] = pd.to_numeric(data['Credit Score'], errors='coerce')
data.loc[data['Credit Score'] > 850, 'Credit Score'] = np.nan

# ── 4. Parse 'Years in current job' ──────────────────────────
data['Years in current job'] = (
    data['Years in current job']
    .astype(str)
    .str.extract(r'(\d+)')
    .astype(float)
)

# ── 5. Use the real Default column ───────────────────────────
print(f"Default distribution:\n{data['Default'].value_counts()}")

# ── 6. Select features ────────────────────────────────────────
feature_cols = [
    'Credit Score', 'Annual Income', 'Monthly Debt', 'Tax Liens',
    'Bankruptcies', 'Number of Credit Problems', 'Years in current job',
    'Number of Open Accounts', 'Years of Credit History',
    'Maximum Open Credit', 'Current Loan Amount', 'Current Credit Balance',
    'Months since last delinquent', 'Home Ownership', 'Purpose', 'Term'
]

features = data[feature_cols].copy()
target = data['Default']

# ── 7. One-hot encode categoricals ───────────────────────────
categorical_columns = ['Home Ownership', 'Purpose', 'Term']
features = pd.get_dummies(features, columns=categorical_columns)

# ── 8. Save feature columns ───────────────────────────────────
os.makedirs('ML_MODEL', exist_ok=True)
feature_columns = features.columns.tolist()
joblib.dump(feature_columns, 'ML_MODEL/feature_columns.pkl')
print(f"Saved {len(feature_columns)} feature columns")

# ── 9. Impute missing values ──────────────────────────────────
imputer = SimpleImputer(strategy='median')
features_imputed = imputer.fit_transform(features)
features_imputed = pd.DataFrame(features_imputed, columns=feature_columns)
joblib.dump(imputer, 'ML_MODEL/imputer.pkl')

# ── 10. Train/test split ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    features_imputed, target, test_size=0.2, random_state=42, stratify=target
)

# ── 11. SMOTE for class imbalance ─────────────────────────────
print(f"\nBefore SMOTE: {y_train.value_counts().to_dict()}")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"After SMOTE:  {pd.Series(y_train_bal).value_counts().to_dict()}")

# ── 12. Define models ─────────────────────────────────────────
models = {
    'knn_model': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier(n_neighbors=5))
    ]),
    'decision_tree_model': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', DecisionTreeClassifier(
            random_state=42, max_depth=10,
            min_samples_split=10, min_samples_leaf=5
        ))
    ]),
    'random_forest_model': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=100, random_state=42,
            max_depth=10, min_samples_split=10, min_samples_leaf=5
        ))
    ]),
    'logistic_regression_model': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, C=0.1))
    ]),
    'naive_bayes_model': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GaussianNB())
    ]),
}

# ── 13. Train, evaluate, save ─────────────────────────────────
print("\n--- Model Results ---")
accuracies = {}
for name, pipeline in models.items():
    pipeline.fit(X_train_bal, y_train_bal)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = round(acc * 100, 2)
    print(f"\n{name}: Test Accuracy = {acc:.2f}")
    print(classification_report(y_test, y_pred))
    joblib.dump(pipeline, f'ML_MODEL/{name}.pkl')
    print(f"Saved ML_MODEL/{name}.pkl")

joblib.dump(accuracies, 'ML_MODEL/model_accuracies.pkl')
print("\nAll models trained and saved successfully!")
print("\nAccuracies:", accuracies)