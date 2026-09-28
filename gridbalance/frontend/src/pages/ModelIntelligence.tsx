import React, { useState, useEffect } from 'react';
import { Cpu, ShieldAlert, Zap, Lock } from 'lucide-react';
import { api } from '../services/api';
import type { ModelInfoResponse } from '../services/api';

export const ModelIntelligence: React.FC = () => {
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);

  useEffect(() => {
    let isMounted = true;
    const fetchModels = async () => {
      try {
        const res = await api.getModelInfo();
        if (isMounted) {
          setModelInfo(res);
        }
      } catch {
        // Ignore error
      }
    };
    fetchModels();
  }, []);

  const sgcc = modelInfo?.sgcc || {
    model_name: "XGBoost Classifier (18-Feature Pruned)",
    feature_count: 18,
    decision_threshold: 0.50,
    test_f1: 0.4014,
    test_pr_auc: 0.4049,
    test_roc_auc: 0.8347,
    test_recall: 0.5498,
    test_precision: 0.3275,
    sealed_test_size: 6355,
    disclaimer: "Metrics evaluated on sealed test partition (15% split, 6,355 customers). Model & threshold frozen."
  };

  const uci = modelInfo?.forecasting || {
    model_name: "XGBoost Regressor (29-Feature)",
    feature_count: 29,
    forecast_horizon: "next_hour",
    test_mae_kwh: 4421.55,
    test_rmse_kwh: 6654.94,
    test_r2: 0.9944,
    test_smape_pct: 2.18,
    sealed_test_size: 4416,
    disclaimer: "Metrics evaluated on sealed chronological test split (July 1 - Dec 31, 2014, 4,416 hours). Model frozen."
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl sm:text-3xl font-bold text-slate-100 font-mono tracking-tight flex items-center gap-3">
          <Cpu className="w-7 h-7 text-emerald-400" />
          MODEL INTELLIGENCE & RESEARCH METRICS
        </h1>
        <p className="text-slate-400 text-sm mt-1">
          Frozen machine learning model specifications, feature architectures, and research benchmark evaluation results.
        </p>
      </div>

      {/* Label Banner */}
      <div className="bg-emerald-950/20 border border-emerald-800/80 p-4 rounded-xl text-xs font-mono text-emerald-300 flex items-center gap-3">
        <Lock className="w-5 h-5 text-emerald-400 shrink-0" />
        <div>
          <span className="font-bold uppercase tracking-wider block">RESEARCH EVALUATION METRICS</span>
          <span className="text-slate-300">
            These performance metrics reflect rigorous evaluations on permanently sealed benchmark test partitions. Both models and decision thresholds are strictly frozen.
          </span>
        </div>
      </div>

      {/* Model Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* SGCC Card */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-amber-500/10 border border-amber-500/30 flex items-center justify-center text-amber-400">
                <ShieldAlert className="w-5 h-5" />
              </div>
              <div>
                <h2 className="font-bold text-slate-100 font-mono">SGCC Tampering Classifier</h2>
                <p className="text-xs text-slate-400 font-mono">{sgcc.model_name}</p>
              </div>
            </div>
            <span className="px-2.5 py-1 rounded bg-slate-800 text-emerald-400 text-xs font-mono border border-slate-700">
              {sgcc.feature_count} Features
            </span>
          </div>

          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3 font-mono text-xs">
              <div className="bg-slate-950 p-3.5 rounded-lg border border-slate-800">
                <div className="text-[10px] text-slate-400 uppercase">Decision Threshold</div>
                <div className="text-lg font-bold text-amber-400 mt-0.5">{sgcc.decision_threshold ?? 0.50}</div>
              </div>
              <div className="bg-slate-950 p-3.5 rounded-lg border border-slate-800">
                <div className="text-[10px] text-slate-400 uppercase">Sealed Test Size</div>
                <div className="text-lg font-bold text-slate-200 mt-0.5">{sgcc.sealed_test_size?.toLocaleString()} customers</div>
              </div>
            </div>

            <div className="bg-slate-950 p-4 rounded-lg border border-slate-800 space-y-3 font-mono">
              <div className="text-xs text-slate-400 font-bold uppercase tracking-wider border-b border-slate-800/80 pb-1.5">
                Classification Benchmark Metrics
              </div>

              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="flex justify-between py-1 border-b border-slate-900">
                  <span className="text-slate-400">Test F1-Score:</span>
                  <span className="font-bold text-slate-100">{sgcc.test_f1}</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-900">
                  <span className="text-slate-400">Test PR-AUC:</span>
                  <span className="font-bold text-slate-100">{sgcc.test_pr_auc}</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-900">
                  <span className="text-slate-400">Test ROC-AUC:</span>
                  <span className="font-bold text-slate-100">{sgcc.test_roc_auc}</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-900">
                  <span className="text-slate-400">Test Recall:</span>
                  <span className="font-bold text-slate-100">{sgcc.test_recall ?? '0.5498'}</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-slate-400">Test Precision:</span>
                  <span className="font-bold text-slate-100">{sgcc.test_precision ?? '0.3275'}</span>
                </div>
              </div>
            </div>

            <p className="text-[11px] text-slate-500 font-mono italic">
              {sgcc.disclaimer}
            </p>
          </div>
        </div>

        {/* UCI Card */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
                <Zap className="w-5 h-5" />
              </div>
              <div>
                <h2 className="font-bold text-slate-100 font-mono">UCI Electricity Forecaster</h2>
                <p className="text-xs text-slate-400 font-mono">{uci.model_name}</p>
              </div>
            </div>
            <span className="px-2.5 py-1 rounded bg-slate-800 text-cyan-400 text-xs font-mono border border-slate-700">
              {uci.feature_count} Features
            </span>
          </div>

          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3 font-mono text-xs">
              <div className="bg-slate-950 p-3.5 rounded-lg border border-slate-800">
                <div className="text-[10px] text-slate-400 uppercase">Forecast Horizon</div>
                <div className="text-base font-bold text-cyan-400 mt-0.5">Next-Hour (t+1 hr)</div>
              </div>
              <div className="bg-slate-950 p-3.5 rounded-lg border border-slate-800">
                <div className="text-[10px] text-slate-400 uppercase">Sealed Test Size</div>
                <div className="text-base font-bold text-slate-200 mt-0.5">{uci.sealed_test_size?.toLocaleString()} hours</div>
              </div>
            </div>

            <div className="bg-slate-950 p-4 rounded-lg border border-slate-800 space-y-3 font-mono">
              <div className="text-xs text-slate-400 font-bold uppercase tracking-wider border-b border-slate-800/80 pb-1.5">
                Regression Benchmark Metrics
              </div>

              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="flex justify-between py-1 border-b border-slate-900">
                  <span className="text-slate-400">Test MAE:</span>
                  <span className="font-bold text-slate-100">{uci.test_mae_kwh?.toLocaleString()} kWh</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-900">
                  <span className="text-slate-400">Test RMSE:</span>
                  <span className="font-bold text-slate-100">{uci.test_rmse_kwh?.toLocaleString()} kWh</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-900">
                  <span className="text-slate-400">Test R² Score:</span>
                  <span className="font-bold text-slate-100">{uci.test_r2}</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-900">
                  <span className="text-slate-400">Test sMAPE:</span>
                  <span className="font-bold text-slate-100">{uci.test_smape_pct}%</span>
                </div>
              </div>
            </div>

            <p className="text-[11px] text-slate-500 font-mono italic">
              {uci.disclaimer}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
