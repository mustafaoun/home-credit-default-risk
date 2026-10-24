"""Serve the trained preprocessing/classifier pipeline with real SHAP explanations."""
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import shap

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]


class LoanFeatures(BaseModel):
    features: Dict[str, Any] = Field(..., min_length=1)


def prepare_input(features, pipeline):
    """Derive notebook features, then enforce the artifact's required schema."""
    frame = pd.DataFrame([features])
    expected = list(pipeline.feature_names_in_)
    if "DEBT_TO_INCOME" in expected and "DEBT_TO_INCOME" not in frame:
        if {"AMT_CREDIT", "AMT_INCOME_TOTAL"} <= set(frame):
            income = pd.to_numeric(frame["AMT_INCOME_TOTAL"], errors="coerce")
            credit = pd.to_numeric(frame["AMT_CREDIT"], errors="coerce")
            frame["DEBT_TO_INCOME"] = (credit / income.replace(0, np.nan)).replace(
                [np.inf, -np.inf], np.nan
            )
    if "DAYS_EMPLOYED_MISSING" in expected and "DAYS_EMPLOYED_MISSING" not in frame:
        if "DAYS_EMPLOYED" in frame:
            frame["DAYS_EMPLOYED_MISSING"] = frame["DAYS_EMPLOYED"].isna().astype(int)
    if "DAYS_EMPLOYED_MISSING" in expected and "DAYS_EMPLOYED" in frame:
        frame["DAYS_EMPLOYED"] = frame["DAYS_EMPLOYED"].fillna(0)
    missing = sorted(set(expected) - set(frame))
    if missing:
        raise HTTPException(422, detail={"message": "Missing model inputs", "fields": missing})
    frame = frame.reindex(columns=expected)
    # JSON nulls must become numeric NaNs for sklearn's fitted numeric imputers.
    preprocessor = pipeline.named_steps["preprocessor"]
    for name, _, columns in preprocessor.transformers_:
        if name == "num":
            for column in columns:
                try:
                    frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
                except (ValueError, TypeError) as exc:
                    raise HTTPException(422, detail=f"Invalid numeric input: {column}") from exc
    return frame


def explanation_vector(explanation, feature_count):
    """Normalize TreeExplainer output across SHAP's supported binary formats."""
    values = explanation.values if hasattr(explanation, "values") else explanation
    if isinstance(values, list):
        values = values[1] if len(values) == 2 else values[0]
    values = np.asarray(values)
    if values.ndim == 3 and values.shape[-1] == 2:
        values = values[:, :, 1]
    if values.ndim == 2 and values.shape[0] == 1:
        values = values[0]
    if values.shape != (feature_count,):
        raise ValueError("SHAP output does not match transformed feature names")
    if not np.isfinite(values).all():
        raise ValueError("SHAP output contains non-finite values")
    return values


def create_app(model_path=None, pipeline_loader=joblib.load, explainer_factory=shap.TreeExplainer):
    @asynccontextmanager
    async def lifespan(application):
        path = Path(model_path or os.getenv("MODEL_PATH", str(ROOT / "models" / "model.pkl")))
        if not path.is_file():
            raise RuntimeError(f"Model not found at {path}. Run python -m src.ml.train first.")
        pipeline = pipeline_loader(path)
        application.state.pipeline = pipeline
        application.state.explainer = explainer_factory(pipeline.named_steps["classifier"])
        threshold = float(os.getenv("DECISION_THRESHOLD", "0.12"))
        if not 0 <= threshold <= 1:
            raise ValueError("DECISION_THRESHOLD must be between 0 and 1")
        application.state.threshold = threshold
        yield

    application = FastAPI(
        title="Home Credit Default Risk API",
        description="Educational credit-risk predictions and model feature contributions.",
        lifespan=lifespan,
    )

    @application.get("/health")
    def health(request: Request):
        loaded = hasattr(request.app.state, "pipeline") and hasattr(request.app.state, "explainer")
        return {"status": "ok" if loaded else "not_ready", "model_loaded": loaded}

    @application.get("/schema")
    def schema(request: Request):
        return {
            "required_model_features": list(request.app.state.pipeline.feature_names_in_),
            "derived_features": ["DEBT_TO_INCOME", "DAYS_EMPLOYED_MISSING"],
            "threshold": request.app.state.threshold,
            "note": "Use the example_request.json generated alongside your model.",
        }

    @application.post("/predict")
    def predict(loan: LoanFeatures, request: Request):
        pipeline = request.app.state.pipeline
        frame = prepare_input(loan.features, pipeline)
        try:
            processed = pipeline.named_steps["preprocessor"].transform(frame)
            probability = float(pipeline.named_steps["classifier"].predict_proba(processed)[0, 1])
            names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
            # XGBoost's TreeExplainer expects a dense feature matrix.
            dense = processed.toarray() if hasattr(processed, "toarray") else processed
            values = explanation_vector(request.app.state.explainer(dense), len(names))
            return {
                "prediction": int(probability >= request.app.state.threshold),
                "probability_default": probability,
                "threshold": request.app.state.threshold,
                "shap_values": dict(zip(names, map(float, values))),
                "shap_units": "model raw output, not percentage-point probability changes",
            }
        except Exception as exc:
            LOGGER.exception("Prediction or explanation failed")
            raise HTTPException(500, detail="Prediction or explanation failed; check server logs.") from exc

    return application


app = create_app()
