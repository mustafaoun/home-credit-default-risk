import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.api.app import create_app, explanation_vector
from src.ml.train import train


@pytest.fixture
def client(tmp_path):
    # Real fitted tree + real TreeExplainer; no Kaggle download or cloud service.
    x = pd.DataFrame({"AMT_CREDIT": [10., 20., 30., 40., 50., 60., 70., 80.],
                      "AMT_INCOME_TOTAL": [100., 90., 80., 70., 60., 50., 40., 30.]})
    pipeline = Pipeline([
        ("preprocessor", ColumnTransformer([("num", SimpleImputer(), list(x))])),
        ("classifier", XGBClassifier(n_estimators=3, max_depth=1, min_child_weight=0,
                                     learning_rate=0.3, base_score=0.5)),
    ])
    pipeline.fit(x, np.array([0, 0, 0, 0, 1, 1, 1, 1]))
    artifact = tmp_path / "model.pkl"
    joblib.dump(pipeline, artifact)
    with TestClient(create_app(artifact)) as value:
        yield value


def test_health_and_real_explanation(client):
    assert client.get("/health").json()["model_loaded"] is True
    response = client.post("/predict", json={"features": {"AMT_CREDIT": 30, "AMT_INCOME_TOTAL": 70}})
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["probability_default"] <= 1
    assert body["prediction"] == int(body["probability_default"] >= body["threshold"])
    assert set(body["shap_values"]) == {"num__AMT_CREDIT", "num__AMT_INCOME_TOTAL"}
    assert all(np.isfinite(v) for v in body["shap_values"].values())


def test_missing_and_invalid_fields_return_validation_errors(client):
    assert client.post("/predict", json={"features": {"AMT_CREDIT": 30}}).status_code == 422
    assert client.post("/predict", json={"features": {
        "AMT_CREDIT": "invalid", "AMT_INCOME_TOTAL": 70}}).status_code == 422
    assert client.post("/predict", json={"features": {}}).status_code == 422


def test_missing_artifact_fails_startup(tmp_path):
    with pytest.raises(RuntimeError, match="Run python -m src.ml.train"):
        with TestClient(create_app(tmp_path / "absent.pkl")):
            pass


def test_schema_matches_loaded_artifact(client):
    assert client.get("/schema").json()["required_model_features"] == [
        "AMT_CREDIT", "AMT_INCOME_TOTAL"
    ]


def test_shap_binary_outputs():
    np.testing.assert_allclose(explanation_vector(np.array([[1., 2.]]), 2), [1, 2])
    np.testing.assert_allclose(explanation_vector(
        [np.array([[-1., -2.]]), np.array([[1., 2.]])], 2), [1, 2])
    with pytest.raises(ValueError):
        explanation_vector(np.zeros((1, 3)), 2)


def test_rebuild_emits_request_that_server_accepts(tmp_path):
    count = 60
    raw = pd.DataFrame({
        "TARGET": [0, 1] * (count // 2),
        "SK_ID_CURR": range(count),
        "DAYS_EMPLOYED": [-100, np.nan, -50] * (count // 3),
        "AMT_CREDIT": np.linspace(10., 200., count),
        "AMT_INCOME_TOTAL": [0, 100, 200] * (count // 3),
        "NAME_EDUCATION_TYPE": ["Higher", "Secondary"] * (count // 2),
    })
    csv = tmp_path / "application_train.csv"
    raw.to_csv(csv, index=False)
    artifacts = tmp_path / "models"
    pipeline = train(csv, artifacts)
    from src.api.app import prepare_input
    from src.ml.train import engineer_features
    raw_request = raw.drop(columns=["TARGET", "SK_ID_CURR"]).iloc[1].to_dict()
    raw_request["DAYS_EMPLOYED"] = None
    expected = engineer_features(pd.DataFrame([raw_request])).reindex(columns=pipeline.feature_names_in_)
    actual = prepare_input(raw_request, pipeline)
    assert actual["DAYS_EMPLOYED"].iloc[0] == expected["DAYS_EMPLOYED"].iloc[0] == 0
    assert actual["DAYS_EMPLOYED_MISSING"].iloc[0] == 1
    body = json.loads((artifacts / "example_request.json").read_text())
    with TestClient(create_app(artifacts / "model.pkl")) as client:
        response = client.post("/predict", json=body)
        assert response.status_code == 200, response.text
        assert isinstance(response.json()["shap_values"], dict)
