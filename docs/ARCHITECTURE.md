# Architecture

`application_train.csv` → raw stratified split → training-only feature selection and preprocessing → XGBoost pipeline → `models/model.pkl`.

FastAPI loads the trusted pipeline at startup and initializes TreeExplainer on its classifier. Requests are validated against the fitted schema, engineered fields are derived, and the same fitted preprocessor feeds probability prediction and SHAP. Missing artifacts fail startup; explanation failures return an error rather than a placeholder explanation.

`/health` reports readiness, `/schema` describes expected inputs, `/predict` returns decisions and explanations. `/docs` provides the generated API schema.

Train locally before `docker build -f deploy/Dockerfile -t home-credit-api .`. See the root README for commands. Monitoring and hardened production deployment are not implemented here.
