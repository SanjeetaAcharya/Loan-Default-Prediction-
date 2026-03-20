from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import io

app = FastAPI(title="Loan Default Prediction API", version="1.0.0")

# ── CORS (allows Node.js to call this API) ────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Node.js URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models once at startup ───────────────────────────────
MODELS = {}
MODEL_FILES = {
    "random_forest":       "ML_MODEL/random_forest_model.pkl",
    "logistic_regression": "ML_MODEL/logistic_regression_model.pkl",
    "decision_tree":       "ML_MODEL/decision_tree_model.pkl",
    "knn":                 "ML_MODEL/knn_model.pkl",
    "naive_bayes":         "ML_MODEL/naive_bayes_model.pkl",
}

try:
    for name, path in MODEL_FILES.items():
        MODELS[name] = joblib.load(path)
    FEATURE_COLUMNS = joblib.load("ML_MODEL/feature_columns.pkl")
    ACCURACIES      = joblib.load("ML_MODEL/model_accuracies.pkl")
    print("✅ All models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")


# ── Input schema ──────────────────────────────────────────────
class PredictionInput(BaseModel):
    model:                    str = "random_forest"
    credit_score:             float
    annual_income:            float
    monthly_debt:             float
    tax_liens:                float
    bankruptcies:             float
    credit_problems:          float
    years_in_job:             float
    open_accounts:            float
    credit_history:           float
    max_open_credit:          float
    loan_amount:              float
    credit_balance:           float
    months_delinquent:        float
    home_ownership:           str
    purpose:                  str
    term:                     str


# ── Helper: preprocess input ──────────────────────────────────
def preprocess(data: dict) -> np.ndarray:
    df = pd.DataFrame([{
        'Credit Score':                 data['credit_score'],
        'Annual Income':                data['annual_income'],
        'Monthly Debt':                 data['monthly_debt'],
        'Tax Liens':                    data['tax_liens'],
        'Bankruptcies':                 data['bankruptcies'],
        'Number of Credit Problems':    data['credit_problems'],
        'Years in current job':         data['years_in_job'],
        'Number of Open Accounts':      data['open_accounts'],
        'Years of Credit History':      data['credit_history'],
        'Maximum Open Credit':          data['max_open_credit'],
        'Current Loan Amount':          data['loan_amount'],
        'Current Credit Balance':       data['credit_balance'],
        'Months since last delinquent': data['months_delinquent'],
        'Home Ownership':               data['home_ownership'],
        'Purpose':                      data['purpose'],
        'Term':                         data['term'],
    }])

    categorical_columns = ['Home Ownership', 'Purpose', 'Term']
    df = pd.get_dummies(df, columns=categorical_columns)
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return df


# ── Routes ────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "Loan Default Prediction API is running"}


@app.get("/models")
def get_models():
    """Returns available models and their accuracies."""
    return {
        "models": list(MODELS.keys()),
        "accuracies": ACCURACIES
    }


@app.post("/predict")
def predict(input: PredictionInput):
    """Single prediction endpoint."""
    if input.model not in MODELS:
        raise HTTPException(status_code=400, detail=f"Model '{input.model}' not found.")

    try:
        pipeline   = MODELS[input.model]
        df         = preprocess(input.dict())
        scaled     = pipeline.named_steps['scaler'].transform(df)
        classifier = pipeline.named_steps['classifier']
        prediction = int(classifier.predict(scaled)[0])
        probability = float(classifier.predict_proba(scaled)[0][1])

        # Risk factors
        risk_factors  = []
        monthly_income = input.annual_income / 12 if input.annual_income > 0 else 1
        dti = input.monthly_debt / monthly_income

        if input.credit_score < 650:
            risk_factors.append(f"Credit score {input.credit_score} is below 650")
        if input.tax_liens > 0:
            risk_factors.append(f"{int(input.tax_liens)} tax lien(s) on record")
        if input.bankruptcies > 0:
            risk_factors.append(f"{int(input.bankruptcies)} bankruptcy record(s)")
        if input.credit_problems > 1:
            risk_factors.append(f"{int(input.credit_problems)} credit problems detected")
        if dti > 0.5:
            risk_factors.append(f"Debt-to-income ratio is {dti:.1%} (above 50%)")

        return {
            "prediction":    prediction,        # 0 or 1
            "probability":   round(probability, 4),
            "risk":          "HIGH" if prediction == 1 else "LOW",
            "risk_factors":  risk_factors,
            "dti_ratio":     round(dti, 4),
            "model_used":    input.model,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(
    file:  UploadFile = File(...),
    model: str        = "random_forest"
):
    """Batch prediction from uploaded CSV file."""
    if model not in MODELS:
        raise HTTPException(status_code=400, detail=f"Model '{model}' not found.")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    try:
        contents = await file.read()
        df_raw   = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        required_cols = [
            'Credit Score', 'Annual Income', 'Monthly Debt', 'Tax Liens',
            'Bankruptcies', 'Number of Credit Problems', 'Years in current job',
            'Number of Open Accounts', 'Years of Credit History',
            'Maximum Open Credit', 'Current Loan Amount', 'Current Credit Balance',
            'Months since last delinquent', 'Home Ownership', 'Purpose', 'Term'
        ]

        missing = [c for c in required_cols if c not in df_raw.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {missing}"
            )

        # Parse years in job
        df_raw['Years in current job'] = (
            df_raw['Years in current job']
            .astype(str).str.extract(r'(\d+)').astype(float)
        )

        pipeline   = MODELS[model]
        results    = []

        for i, row in df_raw.iterrows():
            try:
                processed  = preprocess(dict(
                    credit_score             = row['Credit Score'],
                    annual_income            = row['Annual Income'],
                    monthly_debt             = row['Monthly Debt'],
                    tax_liens                = row['Tax Liens'],
                    bankruptcies             = row['Bankruptcies'],
                    credit_problems          = row['Number of Credit Problems'],
                    years_in_job             = row['Years in current job'],
                    open_accounts            = row['Number of Open Accounts'],
                    credit_history           = row['Years of Credit History'],
                    max_open_credit          = row['Maximum Open Credit'],
                    loan_amount              = row['Current Loan Amount'],
                    credit_balance           = row['Current Credit Balance'],
                    months_delinquent        = row['Months since last delinquent'],
                    home_ownership           = row['Home Ownership'],
                    purpose                  = row['Purpose'],
                    term                     = row['Term'],
                ))
                scaled      = pipeline.named_steps['scaler'].transform(processed)
                classifier  = pipeline.named_steps['classifier']
                prediction  = int(classifier.predict(scaled)[0])
                probability = float(classifier.predict_proba(scaled)[0][1])

                results.append({
                    "row":         i + 1,
                    "prediction":  prediction,
                    "probability": round(probability, 4),
                    "risk":        "HIGH" if prediction == 1 else "LOW",
                })
            except Exception:
                results.append({"row": i + 1, "error": "Could not process this row"})

        total     = len(results)
        high_risk = sum(1 for r in results if r.get("risk") == "HIGH")

        return {
            "total":      total,
            "high_risk":  high_risk,
            "low_risk":   total - high_risk,
            "model_used": model,
            "results":    results,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))