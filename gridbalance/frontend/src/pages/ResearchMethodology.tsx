import React, { useState } from 'react';
import { FileText, ShieldAlert, Zap, Layers, Lock } from 'lucide-react';

export const ResearchMethodology: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'sgcc' | 'uci' | 'robustness'>('sgcc');

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      {/* Header */}
      <div>
        <h1 className="text-2xl sm:text-3xl font-bold text-slate-100 font-mono tracking-tight flex items-center gap-3">
          <FileText className="w-7 h-7 text-emerald-400" />
          RESEARCH METHODOLOGY & SCIENTIFIC AUDIT
        </h1>
        <p className="text-slate-400 text-sm mt-1">
          Complete documentation of data preprocessing, feature engineering, sealed test splits, SHAP interpretability, and robustness testing.
        </p>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-slate-800 font-mono text-xs">
        <button
          onClick={() => setActiveTab('sgcc')}
          className={`px-4 py-3 border-b-2 font-bold transition-colors cursor-pointer flex items-center gap-2 ${
            activeTab === 'sgcc' ? 'border-amber-400 text-amber-400 bg-amber-500/5' : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <ShieldAlert className="w-4 h-4" /> SGCC Tampering Branch
        </button>
        <button
          onClick={() => setActiveTab('uci')}
          className={`px-4 py-3 border-b-2 font-bold transition-colors cursor-pointer flex items-center gap-2 ${
            activeTab === 'uci' ? 'border-cyan-400 text-cyan-400 bg-cyan-500/5' : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <Zap className="w-4 h-4" /> UCI Load Forecasting Branch
        </button>
        <button
          onClick={() => setActiveTab('robustness')}
          className={`px-4 py-3 border-b-2 font-bold transition-colors cursor-pointer flex items-center gap-2 ${
            activeTab === 'robustness' ? 'border-emerald-400 text-emerald-400 bg-emerald-500/5' : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <Layers className="w-4 h-4" /> Robustness & Noise Audit
        </button>
      </div>

      {/* Tab 1: SGCC */}
      {activeTab === 'sgcc' && (
        <div className="space-y-6 text-sm text-slate-300 leading-relaxed font-sans">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-3">
            <h3 className="text-base font-bold font-mono text-slate-100 flex items-center gap-2">
              <Lock className="w-4 h-4 text-amber-400" /> Chronological Date Correction & Bounded Imputation
            </h3>
            <p>
              In the raw SGCC dataset (42,372 customers), date columns were sorted lexicographically rather than chronologically (e.g., 2014/1/1, 2014/1/10 instead of 2014/1/2). This pipeline programmatically parses all column headers into datetime objects and sorts them chronologically prior to any feature extraction.
            </p>
            <p>
              To eliminate reverse temporal leakage caused by unrestricted backward-filling across multi-month communication outages, a bounded forward-fill limit (`ffill(limit=7)`) was established. Gaps exceeding 7 days remain NaNs in raw history and are imputed via a fitted median `SimpleImputer` strictly inside the training pipeline.
            </p>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-3">
            <h3 className="text-base font-bold font-mono text-slate-100">
              18-Feature Reduced Candidate Selection & Stability
            </h3>
            <p>
              Feature selection was conducted using 5-fold Stratified Cross-Validation on the 85% training split. High-collinearity pairs (r &gt; 0.85) were pruned by retaining only the higher SHAP-importance feature. The final 18-feature model achieved:
            </p>
            <ul className="list-disc list-inside space-y-1 font-mono text-xs text-slate-300 pl-2">
              <li>Test F1-Score: 0.4014</li>
              <li>Test PR-AUC: 0.4049</li>
              <li>Test ROC-AUC: 0.8347</li>
              <li>Fixed Decision Threshold: 0.50</li>
            </ul>
          </div>
        </div>
      )}

      {/* Tab 2: UCI */}
      {activeTab === 'uci' && (
        <div className="space-y-6 text-sm text-slate-300 leading-relaxed font-sans">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-3">
            <h3 className="text-base font-bold font-mono text-slate-100 flex items-center gap-2">
              <Zap className="w-4 h-4 text-cyan-400" /> Causal Temporal Split & 29-Feature Architecture
            </h3>
            <p>
              The UCI Electricity Load Diagram dataset covers 370 client load series aggregated into total grid load in kWh. To prevent temporal lookahead leakage, the partition protocol strictly enforces:
            </p>
            <ul className="list-disc list-inside space-y-1 font-mono text-xs text-slate-300 pl-2">
              <li>Training Set: 2011 to 2013-12-31</li>
              <li>Validation Set: 2014-01-01 to 2014-06-30</li>
              <li>Sealed Test Set: 2014-07-01 to 2014-12-31 (4,416 hours)</li>
            </ul>
            <p>
              Features are strictly causal, constructed from 10 historical lags (`load_t` ... `load_t_minus_168`), 10 rolling statistics (`rolling_mean_3` ... `rolling_max_24`), and 9 calendar/cyclic encodings.
            </p>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-3">
            <h3 className="text-base font-bold font-mono text-slate-100">
              Sealed Test Evaluation Metrics
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono text-xs text-center">
              <div className="bg-slate-950 p-3 rounded border border-slate-800">
                <div className="text-slate-500">MAE</div>
                <div className="text-base font-bold text-slate-100">4,421.55 kWh</div>
              </div>
              <div className="bg-slate-950 p-3 rounded border border-slate-800">
                <div className="text-slate-500">RMSE</div>
                <div className="text-base font-bold text-slate-100">6,654.94 kWh</div>
              </div>
              <div className="bg-slate-950 p-3 rounded border border-slate-800">
                <div className="text-slate-500">R² SCORE</div>
                <div className="text-base font-bold text-slate-100">0.9944</div>
              </div>
              <div className="bg-slate-950 p-3 rounded border border-slate-800">
                <div className="text-slate-500">sMAPE</div>
                <div className="text-base font-bold text-slate-100">2.18%</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Tab 3: Robustness */}
      {activeTab === 'robustness' && (
        <div className="space-y-6 text-sm text-slate-300 leading-relaxed font-sans">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-3">
            <h3 className="text-base font-bold font-mono text-slate-100 flex items-center gap-2">
              <Layers className="w-4 h-4 text-emerald-400" /> Noise Perturbation & Missingness Robustness
            </h3>
            <p>
              To evaluate real-world sensor degradation, zero-mean Gaussian noise ({"sigma in {0.01, 0.05, 0.10, 0.20}"}) was injected into input consumption matrices.
            </p>
            <p>
              The 18-feature model maintained high classification stability up to sigma = 0.05 (PR-AUC degradation &le; 0.02). Missingness ablation experiments confirmed that missingness features contribute critical signal during multi-week communication blackouts.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
