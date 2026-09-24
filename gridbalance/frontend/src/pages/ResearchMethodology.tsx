import React from 'react';
import { BookOpen, Database, Lock, Layers, Activity, CheckCircle2 } from 'lucide-react';

export const ResearchMethodology: React.FC = () => {
  return (
    <div className="space-y-8 max-w-5xl mx-auto text-xs sm:text-sm text-slate-300">
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-3">
          <BookOpen className="w-6 h-6 text-teal-400" />
          GridBalance Research & Methodology
        </h1>
        <p className="text-slate-400 mt-1">
          Complete scientific documentation of datasets, temporal validation protocols, sealed test sets, and model architectures.
        </p>
      </div>

      {/* Branch Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="p-6 rounded-xl bg-slate-900 border border-slate-800 space-y-3">
          <div className="flex items-center gap-2 text-teal-400 font-bold text-base">
            <Database className="w-5 h-5" /> Branch 1: SGCC Meter Tampering Detection
          </div>
          <p className="text-slate-400 leading-relaxed">
            Focused on binary classification of customer load profiles (Normal vs Potential Tampering) using real-world smart meter data from State Grid Corporation of China.
          </p>
          <ul className="space-y-1.5 text-slate-300">
            <li>• <strong>Dataset Size</strong>: 42,367 customers across 1,034 daily readings</li>
            <li>• <strong>Class Balance</strong>: 38,755 Normal (91.5%) vs 3,612 Theft (8.5%)</li>
            <li>• <strong>Chronological Sorting</strong>: Header date parsing fixed lexicographical ordering bugs</li>
            <li>• <strong>Missingness Strategy</strong>: Calculated 5 explicit missingness features prior to bounded imputation (limit=7)</li>
          </ul>
        </div>

        <div className="p-6 rounded-xl bg-slate-900 border border-slate-800 space-y-3">
          <div className="flex items-center gap-2 text-teal-400 font-bold text-base">
            <Activity className="w-5 h-5" /> Branch 2: UCI Load Forecasting
          </div>
          <p className="text-slate-400 leading-relaxed">
            Focused on next-hour system electricity load forecasting using the UCI Electricity Load Diagrams dataset.
          </p>
          <ul className="space-y-1.5 text-slate-300">
            <li>• <strong>Dataset Size</strong>: 370 customer meters resampled into 35,065 aggregate system hourly readings</li>
            <li>• <strong>Target</strong>: Load at time t+1 using strictly past observations up to time t</li>
            <li>• <strong>Features</strong>: 29 causal lag, rolling mean/std, and cyclic time features</li>
            <li>• <strong>Validation</strong>: 5-Fold TimeSeriesSplit expanding-window cross-validation</li>
          </ul>
        </div>
      </div>

      {/* Sealed Test Protocol */}
      <div className="p-6 rounded-xl bg-slate-900 border border-slate-800 space-y-4">
        <h2 className="text-base font-bold text-white flex items-center gap-2">
          <Lock className="w-5 h-5 text-rose-400" /> Sealed Test Protocol & Zero Data Leakage
        </h2>
        <p className="leading-relaxed">
          To enforce maximum scientific rigor and prevent data leakage:
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="p-4 rounded-lg bg-slate-950 border border-slate-800 space-y-1">
            <span className="font-bold text-white">SGCC Sealed Test Set (15%)</span>
            <p className="text-slate-400">
              6,356 customer profiles were completely isolated. Feature selection frequency, hyperparameter tuning, and threshold selection were performed strictly on Training/Validation splits.
            </p>
          </div>
          <div className="p-4 rounded-lg bg-slate-950 border border-slate-800 space-y-1">
            <span className="font-bold text-white">UCI Chronological Test Set (12.5%)</span>
            <p className="text-slate-400">
              The final 6-month period (July - Dec 2014, 4,416 hours) was kept sealed until model freezing. Time-series CV ensured zero future lookahead.
            </p>
          </div>
        </div>
      </div>

      {/* Feature Selection & Stability */}
      <div className="p-6 rounded-xl bg-slate-900 border border-slate-800 space-y-4">
        <h2 className="text-base font-bold text-white flex items-center gap-2">
          <Layers className="w-5 h-5 text-sky-400" /> Feature Selection & Robustness Findings
        </h2>
        <p className="leading-relaxed">
          SHAP explainability and 5-fold feature selection frequency analysis reduced the SGCC feature representation from 42 to 18 non-redundant features.
        </p>
        <div className="space-y-2 text-slate-300">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
            <span>Missingness streak count (missing_streak_count) achieved 100% selection stability across 5 training folds.</span>
          </div>
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
            <span>Collinear pairs (correlation &ge; 0.85, e.g. variance vs std_dev) were systematically pruned without degrading classification quality.</span>
          </div>
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
            <span>The 18-feature model demonstrated superior noise resilience (2.05% flip rate at 5% additive noise).</span>
          </div>
        </div>
      </div>
    </div>
  );
};
