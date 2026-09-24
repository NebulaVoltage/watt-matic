import React, { useEffect, useState } from 'react';
import { Cpu, ShieldCheck, LineChart, Info, ShieldAlert, Award } from 'lucide-react';
import { fetchModelInfo, type ModelInfoResponse } from '../services/api';

export const ModelIntelligence: React.FC = () => {
  const [info, setInfo] = useState<ModelInfoResponse | null>(null);

  useEffect(() => {
    fetchModelInfo().then(setInfo).catch(console.error);
  }, []);

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      <div>
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-teal-500/10 border border-teal-500/30 text-teal-400 text-xs font-mono mb-2">
          <Award className="w-3.5 h-3.5" /> BENCHMARK RESEARCH METRICS
        </div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-3">
          <Cpu className="w-6 h-6 text-teal-400" />
          Model Intelligence & Research Evidence
        </h1>
        <p className="text-xs sm:text-sm text-slate-400 mt-1">
          Frozen machine learning model architectures, sealed test evaluation metrics, and robustness benchmarks.
        </p>
      </div>

      {/* Model Performance Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* SGCC Classification Engine */}
        <div className="p-6 rounded-xl bg-slate-900 border border-slate-800 space-y-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div className="flex items-center gap-3">
              <div className="p-2.5 bg-rose-500/10 border border-rose-500/30 rounded-lg">
                <ShieldAlert className="w-5 h-5 text-rose-400" />
              </div>
              <div>
                <h2 className="text-base font-bold text-white">SGCC Meter Tampering Engine</h2>
                <span className="text-xs text-slate-400">18-Feature XGBoost Classifier</span>
              </div>
            </div>
            <span className="px-2.5 py-1 rounded bg-slate-800 border border-slate-700 text-xs font-mono text-teal-400">
              Threshold 0.50
            </span>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
              <span className="text-[11px] text-slate-400 uppercase tracking-wider block">Test F1-Score</span>
              <span className="text-2xl font-bold font-mono text-emerald-400">
                {info?.sgcc?.test_f1 || 0.4014}
              </span>
            </div>

            <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
              <span className="text-[11px] text-slate-400 uppercase tracking-wider block">Test PR-AUC</span>
              <span className="text-2xl font-bold font-mono text-teal-400">
                {info?.sgcc?.test_pr_auc || 0.4049}
              </span>
            </div>

            <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
              <span className="text-[11px] text-slate-400 uppercase tracking-wider block">Test ROC-AUC</span>
              <span className="text-2xl font-bold font-mono text-sky-400">
                {info?.sgcc?.test_roc_auc || 0.8347}
              </span>
            </div>

            <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
              <span className="text-[11px] text-slate-400 uppercase tracking-wider block">Test Recall</span>
              <span className="text-2xl font-bold font-mono text-amber-400">
                {info?.sgcc?.test_recall || 0.5498}
              </span>
            </div>
          </div>

          <div className="text-xs text-slate-400 space-y-1 bg-slate-950/50 p-3 rounded-lg border border-slate-800">
            <span className="font-semibold text-slate-300 block">Key Technical Properties:</span>
            <p>• Evaluated on sealed 6,356 SGCC customer profiles.</p>
            <p>• Explicit missingness feature extraction before bounded imputation.</p>
            <p>• Stratified CV feature selection frequency pruning (18 non-redundant features).</p>
          </div>
        </div>

        {/* UCI Forecasting Engine */}
        <div className="p-6 rounded-xl bg-slate-900 border border-slate-800 space-y-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div className="flex items-center gap-3">
              <div className="p-2.5 bg-teal-500/10 border border-teal-500/30 rounded-lg">
                <LineChart className="w-5 h-5 text-teal-400" />
              </div>
              <div>
                <h2 className="text-base font-bold text-white">UCI Load Forecasting Engine</h2>
                <span className="text-xs text-slate-400">29-Feature XGBoost Regressor</span>
              </div>
            </div>
            <span className="px-2.5 py-1 rounded bg-slate-800 border border-slate-700 text-xs font-mono text-teal-400">
              Next-Hour Horizon
            </span>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
              <span className="text-[11px] text-slate-400 uppercase tracking-wider block">Test R² Score</span>
              <span className="text-2xl font-bold font-mono text-emerald-400">
                {info?.forecasting?.test_r2 || 0.9944}
              </span>
            </div>

            <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
              <span className="text-[11px] text-slate-400 uppercase tracking-wider block">Test MAE</span>
              <span className="text-2xl font-bold font-mono text-teal-400">
                {info?.forecasting?.test_mae?.toLocaleString() || '4,421.55'} <span className="text-xs font-sans">kWh</span>
              </span>
            </div>

            <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
              <span className="text-[11px] text-slate-400 uppercase tracking-wider block">Test RMSE</span>
              <span className="text-2xl font-bold font-mono text-sky-400">
                {info?.forecasting?.test_rmse?.toLocaleString() || '6,654.94'} <span className="text-xs font-sans">kWh</span>
              </span>
            </div>

            <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
              <span className="text-[11px] text-slate-400 uppercase tracking-wider block">Test sMAPE</span>
              <span className="text-2xl font-bold font-mono text-indigo-400">
                {info?.forecasting?.test_smape || 2.18}%
              </span>
            </div>
          </div>

          <div className="text-xs text-slate-400 space-y-1 bg-slate-950/50 p-3 rounded-lg border border-slate-800">
            <span className="font-semibold text-slate-300 block">Key Technical Properties:</span>
            <p>• Evaluated on 4,416-hour chronological test period (July - Dec 2014).</p>
            <p>• 78.75% MAE improvement over Daily Seasonal Naive baseline.</p>
            <p>• Strict causal feature engineering (no future lookahead).</p>
          </div>
        </div>
      </div>

      {/* Robustness Evidence Section */}
      <div className="p-6 rounded-xl bg-slate-900 border border-slate-800 space-y-4">
        <h3 className="text-sm font-semibold text-white flex items-center gap-2">
          <ShieldCheck className="w-4 h-4 text-teal-400" />
          Robustness & Sensitivity Benchmark Summary
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
          <div className="p-4 rounded-lg bg-slate-950 border border-slate-800 space-y-2">
            <span className="font-bold text-teal-400 block">Benign Noise Resilience</span>
            <p className="text-slate-300">
              Under 5% Additive Gaussian measurement noise, the 18-feature model maintained a stable F1 score of **0.4202**, experiencing a minimal prediction flip rate of 2.05%.
            </p>
          </div>

          <div className="p-4 rounded-lg bg-slate-950 border border-slate-800 space-y-2">
            <span className="font-bold text-sky-400 block">Telemetry Data Loss</span>
            <p className="text-slate-300">
              Under long 30% missingness outage bursts, bounded `ffill(limit=7)` imputation preserved a high detection Recall of **52.2%**.
            </p>
          </div>

          <div className="p-4 rounded-lg bg-slate-950 border border-slate-800 space-y-2">
            <span className="font-bold text-amber-400 block">Evasion Sensitivity</span>
            <p className="text-slate-300">
              Evaluated load scaling ($\alpha x_t$) to test detection boundaries. Extreme flatlining reduces recall as zero consumption resembles inactive meters.
            </p>
          </div>
        </div>
      </div>

      {/* Formal Research Disclaimer */}
      <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs flex items-start gap-3">
        <Info className="w-5 h-5 shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold block mb-0.5">Research Evaluation Metric Disclaimer</span>
          All metrics shown on this page represent scientific evaluation results on sealed benchmark datasets (`SGCC` & `UCI`). They are research evaluation metrics and do NOT represent production deployment certifications.
        </div>
      </div>
    </div>
  );
};
