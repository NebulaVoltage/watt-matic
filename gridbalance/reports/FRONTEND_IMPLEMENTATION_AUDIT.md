# GridBalance — Frontend UI Failure Diagnosis & Repair Audit

## 1. Executive Summary

This diagnosis identifies the exact technical root causes behind the broken frontend UI layout and establishes the step-by-step repair strategy to restore the GridBalance visual design system.

---

## 2. Root Cause Analysis

| Component / File | Identified Defect | Impact on Application UI |
| :--- | :--- | :--- |
| `src/index.css` | **Missing Tailwind Import** (`@import "tailwindcss";`) and residual Vite template rules (`#root { width: 1126px; margin: 0 auto; text-align: center; }`, `h1 { font-size: 56px; }`). | All Tailwind utility classes were ignored. The entire application was forced into a 1126px centered box with `text-align: center`, default unstyled buttons, huge 56px headings, and broken layouts. |
| `postcss.config.js` | **0-byte empty file**. `@tailwindcss/postcss` plugin was not registered. | Vite's CSS pipeline could not process Tailwind CSS v4 directives. |
| `index.html` | Default `<title>frontend</title>` header title. | Unprofessional browser tab label. |
| Typography Hierarchy | Heading rules in `index.css` overriding inline Tailwind font sizes. | Section headers (`METER TAMPERING DETECTION`) appeared disproportionately massive and collided with surrounding text. |
| Responsive Layout | Root container enforcing fixed width instead of full-width flex/grid layout (`w-full min-h-screen bg-slate-950`). | Layout collapsed into a single centered column on all screen sizes. |

---

## 3. Repair Strategy

1. **Configure PostCSS & Tailwind v4**:
   - Update `postcss.config.js` to register `@tailwindcss/postcss`.
   - Update `src/index.css` to import `@import "tailwindcss";` and replace Vite template CSS overrides with clean resets.
2. **Design System & Palette Alignment**:
   - Dark technical command-center palette:
     - Background: `#090B10` / `#0D1117` (`bg-slate-950`)
     - Surface Cards: `#11151D` / `#151A23` (`bg-slate-900`, `border-slate-800`)
     - Primary Accents: Cyan (`#22d3ee`), Emerald Green (`#34d399`), Amber (`#fbbf24`), Rose (`#f87171`)
     - Typography: Clean sans-serif headers (24–32px page titles, 16–18px section headers), monospace metrics (12–14px).
3. **Application Shell & Layout**:
   - Full-width dark background (`min-h-screen bg-slate-950 text-slate-100 flex flex-col`).
   - Sticky navbar header with live backend status badge (`● ML Engine Online` / `● ML Engine Offline`).
   - Main content container with `max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 w-full flex-1`.
   - Footer with research evaluation disclaimers and Swagger API documentation links.
4. **Component Refactoring**:
   - Refactor Overview Dashboard (`/`), Detection (`/detection`), Batch Detection (`/detection/batch`), Meter Profile (`/detection/:meterId`), Load Forecasting (`/forecast`), Model Intelligence (`/models`), and Research (`/research`) to enforce card layouts, status strips, input form styling, Recharts visualization, and responsive grids.
