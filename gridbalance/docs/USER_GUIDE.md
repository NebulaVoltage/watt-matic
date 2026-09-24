# GridBalance User Guide

## Getting Started

### 1. Prerequisites
- Python 3.11+
- Node.js v18+ and npm

### 2. Starting the Application

#### A. Backend (FastAPI)
```bash
cd gridbalance
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

#### B. Frontend (React + Vite)
```bash
cd gridbalance/frontend
npm run dev
```

Open browser at `http://127.0.0.1:5173`.

---

## Operating Key Features

### 1. Meter Tampering Detection (`/detection`)
- Enter a meter ID or upload a customer consumption CSV file.
- Click **Analyze Meter Profile**.
- View prediction status (`Normal` or `Potential Tampering`), probability score, risk rating, data quality metrics, SHAP feature contributions, and interactive consumption history.

### 2. Batch Analysis (`/detection/batch`)
- Upload a multi-meter CSV containing customer load trajectories.
- Filter, search, and sort results in real-time.
- Click any meter row to open its detailed intelligence view.
- Export results to CSV.

### 3. Load Forecasting (`/forecast`)
- Upload historical aggregate load CSV or evaluate benchmark data.
- Receive immediate next-hour load predictions ($y_{t+1}$), load delta percentage, and historical trend charts.

### 4. Model Intelligence (`/models`)
- Inspect research benchmark performance, SHAP feature importance rankings, and perturbation robustness results.
