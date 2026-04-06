from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import pandas as pd
import joblib
import shap
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Home Credit Default Risk API", description="Predict loan default probability with SHAP explanations")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model pipeline
model = joblib.load('../model.pkl')

# Precompute background for SHAP (use small sample from training-like data)
# For demo, use a dummy background; in prod, save explainer
# SHAP explainer (lazy init in predict)
explainer = None

class LoanFeatures(BaseModel):
    features: Dict[str, Any]

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}

@app.post("/predict")
def predict(loan: LoanFeatures):
    try:
        df_input = pd.DataFrame([loan.features])
        
        # Preprocess
        X_processed = model.named_steps['preprocessor'].transform(df_input)
        
        # Predict
        pred = model.named_steps['classifier'].predict(X_processed)[0]
        proba = model.named_steps['classifier'].predict_proba(X_processed)[0, 1]
        
        # SHAP (simplified for demo)
        try:
            shap_values = explainer.shap_values(X_processed)
            feature_names = model.named_steps['preprocessor'].get_feature_names_out().tolist()
            shap_vals = shap_values[0].tolist() if isinstance(shap_values, list) else shap_values.values[0].tolist()
            return {
                "prediction": int(pred),
                "probability_default": float(proba),
                "shap_values": dict(zip(feature_names[:len(shap_vals)], shap_vals))
            }
        except:
            return {
                "prediction": int(pred),
                "probability_default": float(proba),
                "shap_values": "SHAP computation failed - use more features for full explainer"
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

