# GridBalance Platform Architecture

## 1. System Topology & Layers

```
                      GRIDBALANCE SMART GRID PLATFORM
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           │                                                   │
   [SGCC Detection Engine]                           [UCI Forecasting Engine]
  18-Feature XGBoost Classifier                        XGBoost Regressor
  Threshold: 0.50 (Fixed)                              Horizon: Next-Hour ($y_{t+1}$)
           │                                                   │
           └─────────────────────────┬─────────────────────────┘
                                     │
                             FastAPI REST Backend
                             (Port 8000 / Python 3.11)
                                     │
                             React + TypeScript + Vite
                             (Port 5173 / Tailwind CSS)
                                     │
                           Control Intelligence Dashboard
```

## 2. Component Design & Responsibility
- **FastAPI REST Backend (`backend/`)**:
  - `backend/main.py`: Entry point, CORS middleware, API route mounting.
  - `backend/services/sgcc_service.py`: Loads frozen 18-feature XGBoost model ONCE at startup. Executes feature extraction, bounded imputation (`ffill(limit=7)`), and risk scoring.
  - `backend/services/forecasting_service.py`: Loads frozen UCI XGBoost Regressor ONCE at startup. Executes causal lag and rolling window feature extraction.
  - `backend/schemas/`: Pydantic schemas enforcing input validation and type safety.

- **React + TypeScript Frontend (`frontend/`)**:
  - Built with Vite, Tailwind CSS, and Recharts.
  - Single Page Application (SPA) architecture with client-side routing (`/`, `/detection`, `/detection/batch`, `/detection/:meterId`, `/forecast`, `/models`, `/research`).

## 3. Data Integrity & Safety Principles
- **Frozen Models**: Models are strictly loaded from joblib artifacts. Zero retraining or hyperparameter fitting during runtime requests.
- **Causal Time-Series Rules**: Forecasting lag & rolling features use `shift(1)` causal windows; lookahead is prevented.
- **Terminology Enforcement**: Uses `"Potential Tampering"` instead of `"Confirmed Theft"` to reflect statistical prediction boundaries.
