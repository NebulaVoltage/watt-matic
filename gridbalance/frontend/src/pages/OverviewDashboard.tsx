import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Search, Zap, Activity, ShieldAlert, ArrowRight, Info } from 'lucide-react';
import { api } from '../services/api';
import type { HealthResponse, ModelInfoResponse } from '../services/api';

export const OverviewDashboard: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);

  // Read active session stats if any analysis was run during this browser session
  const [sessionAnalyzed] = useState<number>(() => {
    return parseInt(sessionStorage.getItem('gridbalance_analyzed_count') || '0', 10);
  });
  const [sessionFlagged] = useState<number>(() => {
    return parseInt(sessionStorage.getItem('gridbalance_flagged_count') || '0', 10);
  });
  const [lastForecast] = useState<string | null>(() => {
    return sessionStorage.getItem('gridbalance_last_forecast') || null;
  });

  useEffect(() => {
    let isMounted = true;
    const loadData = async () => {
      try {
        const [hRes, mRes] = await Promise.all([
          api.checkHealth(),
          api.getModelInfo().catch(() => null)
        ]);
        if (isMounted) {
          setHealth(hRes);
          if (mRes) setModelInfo(mRes);
        }
      } catch {
        // Ignore
      }
    };
    loadData();
  }, []);

  const isSgccOnline = health?.sgcc_model === 'loaded';
  const isForecastOnline = health?.forecast_model === 'loaded';

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6 sm:p-10 relative overflow-hidden">
        <div className="absolute -right-10 -top-10 w-80 h-80 bg-emerald-500/5 rounded-full blur-3xl pointer-events-none"></div>
        <div className="absolute -left-10 -bottom-10 w-80 h-80 bg-cyan-500/5 rounded-full blur-3xl pointer-events-none"></div>
        
        <div className="relative z-10 max-w-3xl space-y-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-mono">
            <Zap className="w-3.5 h-3.5" />
            RESEARCH-VALIDATED MACHINE LEARNING PLATFORM
          </div>
          
          <h1 className="text-3xl sm:text-5xl font-extrabold text-slate-100 tracking-tight">
            GRIDBALANCE
          </h1>
          <p className="text-lg sm:text-xl text-emerald-400 font-mono font-medium">
            AI-POWERED SMART GRID INTELLIGENCE
          </p>
          <p className="text-slate-300 text-base leading-relaxed">
            Analyze electricity consumption patterns for potential meter tampering and forecast demand using research-validated machine learning.
          </p>

          <div className="flex flex-wrap items-center gap-4 pt-4">
            <Link
              to="/detection"
              className="px-5 py-2.5 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-semibold text-sm font-mono tracking-wide uppercase transition-all shadow-md hover:shadow-emerald-500/20 flex items-center gap-2"
            >
              <Search className="w-4 h-4" />
              Analyze Meter
            </Link>

            <Link
              to="/forecast"
              className="px-5 py-2.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-cyan-300 border border-cyan-500/30 hover:border-cyan-400 font-semibold text-sm font-mono tracking-wide uppercase transition-all flex items-center gap-2"
            >
              <Zap className="w-4 h-4" />
              Forecast Load
            </Link>
          </div>
        </div>
      </div>

      {/* System Status Cards — Strict Empty States */}
      <div>
        <h2 className="text-sm font-mono uppercase tracking-widest text-slate-400 mb-4 flex items-center gap-2">
          <Activity className="w-4 h-4 text-emerald-400" />
          Live Session Telemetry & Status
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Status Card 1 */}
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-5">
            <div className="text-slate-400 text-xs font-mono mb-2 uppercase">Meters Analyzed</div>
            <div className="text-2xl font-bold font-mono text-slate-100">
              {sessionAnalyzed > 0 ? sessionAnalyzed : '—'}
            </div>
            <p className="text-xs text-slate-500 mt-2 font-mono">
              {sessionAnalyzed > 0 ? `${sessionAnalyzed} meters evaluated in session` : 'No analysis performed yet'}
            </p>
          </div>

          {/* Status Card 2 */}
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-5">
            <div className="text-slate-400 text-xs font-mono mb-2 uppercase">Potential Tampering</div>
            <div className="text-2xl font-bold font-mono text-amber-400">
              {sessionAnalyzed > 0 ? sessionFlagged : '—'}
            </div>
            <p className="text-xs text-slate-500 mt-2 font-mono">
              {sessionAnalyzed > 0 ? `${sessionFlagged} flagged above 0.50 threshold` : 'Awaiting dataset analysis'}
            </p>
          </div>

          {/* Status Card 3 */}
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-5">
            <div className="text-slate-400 text-xs font-mono mb-2 uppercase">Next-Hour Forecast</div>
            <div className="text-2xl font-bold font-mono text-cyan-400">
              {lastForecast ? `${lastForecast} kWh` : '—'}
            </div>
            <p className="text-xs text-slate-500 mt-2 font-mono">
              {lastForecast ? 'Latest t+1 load prediction' : 'No forecast generated yet'}
            </p>
          </div>

          {/* Status Card 4: Real Model Status */}
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-5 flex flex-col justify-between">
            <div className="text-slate-400 text-xs font-mono mb-2 uppercase">Live Engine Health</div>
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-slate-300">SGCC Detection:</span>
                <span className={`flex items-center gap-1 font-semibold ${isSgccOnline ? 'text-emerald-400' : 'text-rose-400'}`}>
                  <span className={`w-2 h-2 rounded-full ${isSgccOnline ? 'bg-emerald-400' : 'bg-rose-500'}`}></span>
                  {isSgccOnline ? 'Online' : 'Offline'}
                </span>
              </div>
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-slate-300">UCI Forecaster:</span>
                <span className={`flex items-center gap-1 font-semibold ${isForecastOnline ? 'text-emerald-400' : 'text-rose-400'}`}>
                  <span className={`w-2 h-2 rounded-full ${isForecastOnline ? 'bg-emerald-400' : 'bg-rose-500'}`}></span>
                  {isForecastOnline ? 'Online' : 'Offline'}
                </span>
              </div>
            </div>
            <p className="text-[11px] text-slate-500 mt-2 font-mono">Sourced directly from FastAPI /health</p>
          </div>
        </div>
      </div>

      {/* Frozen Research Engines Summary */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* SGCC Card */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 hover:border-slate-700 transition-colors">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-amber-500/10 border border-amber-500/30 flex items-center justify-center text-amber-400">
                <ShieldAlert className="w-5 h-5" />
              </div>
              <div>
                <h3 className="font-bold text-slate-100">SGCC Tampering Detection</h3>
                <p className="text-xs text-slate-400 font-mono">Frozen 18-Feature XGBoost Classifier</p>
              </div>
            </div>
            <span className="px-2.5 py-1 rounded bg-slate-800 text-slate-300 text-xs font-mono">Threshold 0.50</span>
          </div>

          <div className="grid grid-cols-3 gap-3 bg-slate-950/60 p-3 rounded-lg border border-slate-800/80 mb-4 font-mono text-center">
            <div>
              <div className="text-[10px] text-slate-400 uppercase">Test F1</div>
              <div className="text-base font-bold text-slate-100">{modelInfo?.sgcc?.test_f1 ?? '0.4014'}</div>
            </div>
            <div>
              <div className="text-[10px] text-slate-400 uppercase">PR-AUC</div>
              <div className="text-base font-bold text-slate-100">{modelInfo?.sgcc?.test_pr_auc ?? '0.4049'}</div>
            </div>
            <div>
              <div className="text-[10px] text-slate-400 uppercase">ROC-AUC</div>
              <div className="text-base font-bold text-slate-100">{modelInfo?.sgcc?.test_roc_auc ?? '0.8347'}</div>
            </div>
          </div>

          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-400 flex items-center gap-1.5">
              <Info className="w-3.5 h-3.5 text-slate-500" />
              Evaluated on 6,355 sealed test customers
            </span>
            <Link to="/detection" className="text-emerald-400 hover:text-emerald-300 font-mono font-semibold flex items-center gap-1">
              Launch Detection <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        </div>

        {/* UCI Card */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 hover:border-slate-700 transition-colors">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
                <Zap className="w-5 h-5" />
              </div>
              <div>
                <h3 className="font-bold text-slate-100">UCI Electricity Load Forecasting</h3>
                <p className="text-xs text-slate-400 font-mono">Frozen 29-Feature XGBoost Regressor</p>
              </div>
            </div>
            <span className="px-2.5 py-1 rounded bg-slate-800 text-slate-300 text-xs font-mono">Horizon t+1 hr</span>
          </div>

          <div className="grid grid-cols-3 gap-3 bg-slate-950/60 p-3 rounded-lg border border-slate-800/80 mb-4 font-mono text-center">
            <div>
              <div className="text-[10px] text-slate-400 uppercase">Test MAE</div>
              <div className="text-base font-bold text-slate-100">4,421 kWh</div>
            </div>
            <div>
              <div className="text-[10px] text-slate-400 uppercase">sMAPE</div>
              <div className="text-base font-bold text-slate-100">2.18%</div>
            </div>
            <div>
              <div className="text-[10px] text-slate-400 uppercase">Test R²</div>
              <div className="text-base font-bold text-slate-100">0.9944</div>
            </div>
          </div>

          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-400 flex items-center gap-1.5">
              <Info className="w-3.5 h-3.5 text-slate-500" />
              Evaluated on 4,416 sealed test hours
            </span>
            <Link to="/forecast" className="text-cyan-400 hover:text-cyan-300 font-mono font-semibold flex items-center gap-1">
              Launch Forecaster <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};
