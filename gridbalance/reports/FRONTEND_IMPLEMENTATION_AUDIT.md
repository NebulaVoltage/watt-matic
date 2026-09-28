# GridBalance — Frontend Implementation Audit & Architecture Plan

## 1. Executive Summary

This audit establishes the frontend technical specifications and component architecture for the **GridBalance Smart Grid Intelligence Web Application**. The application connects directly to the frozen FastAPI inference backend (`http://127.0.0.1:8000/api/v1`) to provide production-grade, research-validated smart grid analytics without mock predictions or artificial statistics.

---

## 2. Technical Stack & Environment Audit

| Layer | Library / Technology | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Framework** | React + TypeScript | React 19.2, TS 6.0 | Component hierarchy, type-safe application logic |
| **Build Tool** | Vite | 8.3.0 | Hot Module Replacement (HMR) and fast production bundle generation |
| **Routing** | React Router DOM | 7.18.4 | Single Page Application (SPA) client-side routing |
| **Styling** | Tailwind CSS v4 + `@tailwindcss/postcss` | 4.3.3 | Dark technical design system (energy/grid aesthetic) |
| **Icons** | Lucide React | 1.48.0 | Clean vector UI iconography |
| **Visualization** | Recharts | 3.10.1 | Interactive time-series consumption & load forecasting charts |
| **API Client** | Native Fetch API | Built-in | Centralized typed HTTP service (`src/services/api.ts`) |

---

## 3. Application Route Map

The application enforces a 7-route navigation model:

1. `/` — **Overview Dashboard**: High-level grid load metrics, active model health indicator (`● ML Engine Online`), real recent session summaries, and quick action launchpad.
2. `/detection` — **Meter Tampering Detection**: Drag-and-drop CSV / form upload for single-meter daily consumption series, invoking `POST /api/v1/detection/analyze`.
3. `/detection/batch` — **Batch Meter Analysis**: Bulk CSV upload invoking `POST /api/v1/detection/batch`, returning searchable, sortable, paginated risk tables with CSV export.
4. `/detection/:meterId` — **Individual Meter Intelligence**: Session-backed deep-dive profile for a specific customer meter, displaying historical consumption trends and data quality signals.
5. `/forecast` — **Electricity Load Forecasting**: Historical hourly load series input ($\ge 168$ hours), invoking `POST /api/v1/forecast/next-hour` to display next-hour ($t+1$) demand predictions.
6. `/models` — **Model Intelligence**: Research evaluation metric dashboards for SGCC (18 features, threshold 0.50) and UCI (29 features, $t+1$ horizon) sourced directly from `GET /api/v1/models`.
7. `/research` — **Research & Methodology**: Scientific documentation of temporal splits, sealed test protocols, missingness ablation, and robustness testing.

---

## 4. Visual Design System

- **Color Palette**:
  - Backgrounds: `slate-950` (`#020617`), `slate-900` (`#0f172a`), `slate-800/80` (`#1e293b`)
  - Typography: Primary `slate-100` (`#f8fafc`), Secondary `slate-400` (`#94a3b8`)
  - Status Accents:
    - Normal / Healthy: `emerald-400` (`#34d399`) / `emerald-500/20`
    - Medium Risk / Warning: `amber-400` (`#fbbf24`) / `amber-500/20`
    - High Risk / Potential Tampering: `rose-400` (`#f87171`) / `rose-500/20`
    - Forecasting / Cyan Accent: `cyan-400` (`#22d3ee`) / `cyan-500/20`
- **Typography & Layout**: Monospace indicators (`font-mono`) for numerical values, probabilities, timestamps, and model parameters; clean sans-serif for headers and body copy.
- **Micro-Interactions**: Subtle border glows (`border-slate-800 hover:border-slate-700`), responsive grid cards, and smooth CSS transitions.

---

## 5. Backend Service Integration Strategy

All frontend pages consume endpoints from `src/services/api.ts`:
- Health Status (`GET /api/v1/health`): Polled on mount and polled periodically to drive the header status badge (`● ML Engine Online` / `● ML Engine Offline`).
- Model Metadata (`GET /api/v1/models`): Cached at app level to populate research benchmark cards.
- Single Meter Inference (`POST /api/v1/detection/analyze`): Formatted payload with dates and consumption array.
- Batch Inference (`POST /api/v1/detection/batch`): `FormData` multipart file upload.
- Next-Hour Forecasting (`POST /api/v1/forecast/next-hour`): Sequential load array payload.

---

## 6. Verification Protocol

The application implementation will be verified by running the live FastAPI server (`127.0.0.1:8000`), launching Vite dev server (`127.0.0.1:5173`), performing real single/batch detection and load forecasting requests, and confirming 100% of data rendered on screen originates from real FastAPI responses.
