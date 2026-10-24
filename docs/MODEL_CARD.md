# Model card

## Purpose

Educational binary prediction of loan default using Home Credit application records. Not approved for real lending, eligibility, pricing, or adverse-action decisions.

## Data

Home Credit Default Risk competition, `application_train.csv`; target `TARGET`, applicant identifier `SK_ID_CURR` excluded. Obtain the dataset under the competition terms. Data provenance and demographic coverage limit generalization.

## Method

Stratified 80/20 train/holdout split, seed 42. Training-only missingness and category selection; median/most-frequent imputation, robust numeric scaling, categorical one-hot encoding, XGBoost. Employment-missingness and debt-to-income features are derived consistently in training and serving. The holdout is used to report ROC-AUC, not to fit preprocessing.

## Evaluation

Original notebook ROC-AUC: 0.7466 (historical, not revalidated by the API tests). Fresh training writes actual evaluation results to `models/metrics.json`. Synthetic regression tests establish functional behavior only. No external validation, subgroup fairness study, calibrated-probability assessment, load test, or prospective financial impact study is supplied.

## Decisions and explanations

Probability threshold defaults to 0.12 and is configurable. That cutoff is illustrative and requires independent validation. SHAP explains the raw classifier score; contributions are not causal effects and do not establish fairness or legal compliance.

## Limitations

Distribution shift, class imbalance, correlated inputs, sample selection, and proxy demographic variables may affect reliability. Earlier financial savings estimates are unvalidated scenarios, not realized results. Deployment requires independent risk assessment, monitoring, security, and human governance.
