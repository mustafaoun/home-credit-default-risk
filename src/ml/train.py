"""Rebuild the model artifact and a complete API request from locally downloaded data."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]


def engineer_features(frame):
    frame = frame.copy()
    frame["DAYS_EMPLOYED_MISSING"] = frame["DAYS_EMPLOYED"].isna().astype(int)
    frame["DAYS_EMPLOYED"] = frame["DAYS_EMPLOYED"].fillna(0)
    income = frame["AMT_INCOME_TOTAL"].replace(0, np.nan)
    frame["DEBT_TO_INCOME"] = (frame["AMT_CREDIT"] / income).replace([np.inf, -np.inf], np.nan)
    return frame


def train(data_path, output_dir):
    raw = pd.read_csv(data_path)
    target = raw.pop("TARGET")
    raw = raw.drop(columns=["SK_ID_CURR"], errors="ignore")
    # Fit data-dependent feature selection using training data only.
    train_raw, test_raw, y_train, y_test = train_test_split(
        raw, target, test_size=0.2, stratify=target, random_state=42
    )
    x_train, x_test = engineer_features(train_raw), engineer_features(test_raw)
    # Keep fields used in feature derivation even when sparse.
    derived_inputs = {"DAYS_EMPLOYED", "AMT_INCOME_TOTAL", "AMT_CREDIT"}
    columns = [c for c in x_train if x_train[c].isna().mean() <= 0.5 or c in derived_inputs]
    x_train, x_test = x_train[columns], x_test[columns]
    numeric = x_train.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in x_train.select_dtypes(include=["object"]) if x_train[c].nunique() < 10]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                           ("scaler", RobustScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                           ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                     random_state=42, eval_metric="logloss")),
    ])
    pipeline.fit(x_train, y_train)
    score = float(roc_auc_score(y_test, pipeline.predict_proba(x_test)[:, 1]))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_dir / "model.pkl")
    # Derivation inputs plus model features make a request that matches this artifact.
    example = json.loads(x_test.iloc[:1].to_json(orient="records"))[0]
    (output_dir / "example_request.json").write_text(json.dumps({"features": example}, indent=2))
    (output_dir / "metrics.json").write_text(json.dumps({
        "roc_auc": score, "test_rows": len(y_test), "split": "stratified 80/20", "random_state": 42,
        "note": "Fresh rebuild result; do not assume it equals the historical notebook metric.",
    }, indent=2))
    print(f"Rebuilt held-out ROC-AUC: {score:.4f}. Artifacts saved to {output_dir}")
    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/application_train.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Download application_train.csv from Home Credit on Kaggle to {args.data}")
    train(args.data, args.output_dir)
