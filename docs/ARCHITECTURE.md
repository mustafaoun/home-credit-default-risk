# Project Architecture

## Folder Structure

```
home-credit-default-risk/
├── notebooks/W2.ipynb          # Exploration, analysis, SHAP
├── src/                        # Production code
│   ├── api/app.py              # FastAPI endpoints
│   └── ml/train.py             # Training pipeline
├── configs/config.yaml         # Optional: Model & API configuration
├── data/                       # Datasets (raw/, processed/)
├── models/                     # Trained artifacts (model.pkl)
├── results/                    # Outputs & visualizations
├── docs/                       # Documentation
├── deploy/Dockerfile           # Docker image
├── tests/                      # Unit tests (optional)
└── deploy/                     # Docker & Render config
```

## Key Directories

| Folder | Purpose |
|--------|---------|
| **notebooks/** | Exploratory analysis (Jupyter) |
| **src/api/** | FastAPI server & endpoints |
| **src/ml/** | Training & preprocessing code |
| **data/** | Raw and processed datasets |
| **models/** | Trained model pickles |
| **configs/** | Optional YAML configuration |
| **results/** | Model outputs & plots |
| **docs/** | Documentation (MODEL_CARD.md, etc.) |
| **deploy/** | Docker & deployment configs |

## Data Flow

```
notebooks/W2.ipynb  →  src/ml/train.py  →  models/model.pkl  →  src/api/app.py
(Explore)            (Train)              (Save)               (Serve)
```

## Notable Decisions

- **One model file**: `model.pkl` includes preprocessor + classifier (single serialize)
- **No configs needed**: Hardcoded values in code are fine for simple projects; use `configs/config.yaml` only if you need multiple environments (dev/prod)
- **Results folder**: Optional; store predictions, metrics, plots here for post-analysis
- **Tests optional**: Add when code complexity warrants it

## How to Use This Structure

1. **Develop**: Write exploration in `notebooks/W2.ipynb`
2. **Clean up**: Move validated code to `src/ml/` and `src/api/`
3. **Train**: Run `python src/ml/train.py` → saves to `models/`
4. **Deploy**: `docker build -t w2-model . && docker run -p 8000:8000 w2-model`

---

For more details on how to structure your code, see README.md and docs/MODEL_CARD.md

