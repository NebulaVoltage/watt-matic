import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Search, Zap, Activity, ShieldAlert, ArrowRight, Info, Cpu, Database, Server, Layout } from 'lucide-react';
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
        // Ignore errors
      }
    };
    loadData();
  }, []);

  const isSgccOnline = health?.sgcc_model === 'loaded';
  const isForecastOnline = health?.forecast_model === 'loaded';

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 sm:p-8 relative overflow-hidden">
        <div className="absolute -right-10 -top-10 w-72 h-72 bg-cyan-500/5 rounded-full blur-3xl pointer-events-none"></div>
        <div className="absolute -left-10 -bottom-10 w-72 h-72 bg-emerald-500/5 rounded-full blur-3xl pointer-events-none"></div>
        
        <div className="relative z-10 max-w-3xl space-y-4 text-left">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-mono">
            <Zap className="w-3.5 h-3.5" />
            RESEARCH-VALIDATED SMART GRID INTELLIGENCE PLATFORM
          </div>
          
          <h1 className="text-2xl sm:text-4xl font-extrabold text-slate-100 tracking-tight font-mono">
            GRIDBALANCE
          </h1>
          <p className="text-base sm:text-lg text-emerald-400 font-mono font-medium">
            AI-DRIVEN SMART GRID INTELLIGENCE & LOAD FORECASTING
          </p>
          <p className="text-slate-300 text-sm leading-relaxed">
            Execute real-time meter tampering detection and next-hour electricity demand forecasting using research-validated, frozen XGBoost models connected to a FastAPI inference backend.
          </p>

          <div className="flex flex-wrap items-center gap-3 pt-2">
            <Link
              to="/detection"
              className="px-4 py-2 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-semibold text-xs font-mono tracking-wide uppercase transition-all shadow-md flex items-center gap-2"
            >
              <Search className="w-3.5 h-3.5" />
              Analyze Meter
            </Link>

            <Link
              to="/forecast"
              className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-cyan-300 border border-cyan-500/30 font-semibold text-xs font-mono tracking-wide uppercase transition-all flex items-center gap-2"
            >
              <Zap className="w-3.5 h-3.5" />
              Forecast Load
            </Link>
          </div>
        </div>
      </div>

      {/* System Status Cards — Strict Empty States */}
      <div>
        <h2 className="text-xs font-mono uppercase tracking-widest text-slate-400 mb-3 flex items-center gap-2">
          <Activity className="w-4 h-4 text-emerald-400" />
          Live Session Telemetry & Status
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
            <div className="text-slate-400 text-xs font-mono mb-1 uppercase">Meters Analyzed</div>
            <div className="text-2xl font-bold font-mono text-slate-100">
              {sessionAnalyzed > 0 ? sessionAnalyzed : '—'}
            </div>
            <p className="text-[11px] text-slate-500 mt-1 font-mono">
              {sessionAnalyzed > 0 ? `${sessionAnalyzed} meters evaluated` : 'No analysis performed yet'}
            </p>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
            <div className="text-slate-400 text-xs font-mono mb-1 uppercase">Potential Tampering</div>
            <div className="text-2xl font-bold font-mono text-amber-400">
              {sessionAnalyzed > 0 ? sessionFlagged : '—'}
            </div>
            <p className="text-[11px] text-slate-500 mt-1 font-mono">
              {sessionAnalyzed > 0 ? `${sessionFlagged} flagged above 0.50 threshold` : 'Awaiting data analysis'}
            </p>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
            <div className="text-slate-400 text-xs font-mono mb-1 uppercase">Next-Hour Forecast</div>
            <div className="text-2xl font-bold font-mono text-cyan-400">
              {lastForecast ? `${lastForecast} kWh` : '—'}
            </div>
            <p className="text-[11px] text-slate-500 mt-1 font-mono">
              {lastForecast ? 'Latest t+1 load prediction' : 'No forecast generated yet'}
            </p>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 flex flex-col justify-between">
            <div className="text-slate-400 text-xs font-mono mb-1 uppercase">Engine Status</div>
            <div className="space-y-1 font-mono text-xs">
              <div className="flex justify-between">
                <span className="text-slate-400">SGCC Detection:</span>
                <span className={`font-bold ${isSgccOnline ? 'text-emerald-400' : 'text-rose-400'}`}>
                  {isSgccOnline ? '● Online' : '● Offline'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">UCI Forecaster:</span>
                <span className={`font-bold ${isForecastOnline ? 'text-emerald-400' : 'text-rose-400'}`}>
                  {isForecastOnline ? '● Online' : '● Offline'}
                </span>
              </div>
            </div>
            <p className="text-[10px] text-slate-500 mt-1 font-mono">GET /api/v1/health</p>
          </div>
        </div>
      </div>

      {/* System Architecture Diagram (Phase 12 Requirement) */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
        <h2 className="text-xs font-mono uppercase tracking-widest text-slate-400 flex items-center gap-2">
          <Server className="w-4 h-4 text-emerald-400" />
          System Architecture Pipeline
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-5 gap-2 text-center font-mono text-xs">
          <div className="bg-slate-950 p-3 rounded border border-slate-800 flex flex-col items-center justify-center gap-1">
            <Database className="w-5 h-5 text-slate-400" />
            <span className="font-bold text-slate-200">Consumption Data</span>
            <span className="text-[10px] text-slate-500">Raw Load Series</span>
          </div>
          <div className="hidden sm:flex items-center justify-center text-slate-600">→</div>

          <div className="bg-slate-950 p-3 rounded border border-slate-800 flex flex-col items-center justify-center gap-1">
            <Cpu className="w-5 h-5 text-emerald-400" />
            <span className="font-bold text-slate-200">18 / 29 Feature Eng</span>
            <span className="text-[10px] text-slate-500">Causal & Gap Metrics</span>
          </div>
          <div className="hidden sm:flex items-center justify-center text-slate-600">→</div>

          <div className="bg-slate-950 p-3 rounded border border-slate-800 flex flex-col items-center justify-center gap-1">
            <Server className="w-5 h-5 text-cyan-400" />
            <span className="font-bold text-slate-200">FastAPI Backend</span>
            <span className="text-[10px] text-slate-500">Frozen Joblib Models</span>
          </div>
          <div className="hidden sm:flex items-center justify-center text-slate-600">→</div>

          <div className="bg-slate-950 p-3 rounded border border-slate-800 flex flex-col items-center justify-center gap-1">
            <Layout className="w-5 h-5 text-amber-400" />
            <span className="font-bold text-slate-200">React Dashboard</span>
            <span className="text-[10px] text-slate-500">Interactive UI</span>
          </div>
        </div>
      </div>

      {/* Frozen Research Engines Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 hover:border-slate-700 transition-colors">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded bg-amber-500/10 border border-amber-500/30 flex items-center justify-center text-amber-400">
                <ShieldAlert className="w-4 h-4" />
              </div>
              <div>
                <h3 className="font-bold text-slate-100 font-mono text-sm">SGCC Tampering Classifier</h3>
                <p className="text-xs text-slate-400 font-mono">Frozen 18-Feature XGBoost</p>
              </div>
            </div>
            <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 text-xs font-mono">Threshold 0.50</span>
          </div>

          <div className="grid grid-cols-3 gap-2 bg-slate-950 p-3 rounded border border-slate-800 text-center font-mono text-xs mb-4">
            <div>
              <div className="text-[10px] text-slate-500">TEST F1</div>
              <div className="font-bold text-slate-200">{modelInfo?.sgcc?.test_f1 ?? '0.4014'}</div>
            </div>
            <div>
              <div className="text-[10px] text-slate-500">PR-AUC</div>
              <div className="font-bold text-slate-200">{modelInfo?.sgcc?.test_pr_auc ?? '0.4049'}</div>
            </div>
            <div>
              <div className="text-[10px] text-slate-500">ROC-AUC</div>
              <div className="font-bold text-slate-200">{modelInfo?.sgcc?.test_roc_auc ?? '0.8347'}</div>
            </div>
          </div>

          <div className="flex items-center justify-between text-xs font-mono">
            <span className="text-slate-500 flex items-center gap-1">
              <Info className="w-3.5 h-3.5" /> 6,355 sealed test customers
            </span>
            <Link to="/detection" className="text-emerald-400 hover:text-emerald-300 font-bold flex items-center gap-1">
              Launch Detection <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 hover:border-slate-700 transition-colors">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
                <Zap className="w-4 h-4" />
              </div>
              <div>
                <h3 className="font-bold text-slate-100 font-mono text-sm">UCI Electricity Forecaster</h3>
                <p className="text-xs text-slate-400 font-mono">Frozen 29-Feature XGBoost Regressor</p>
              </div>
            </div>
            <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 text-xs font-mono">Horizon t+1 hr</span>
          </div>

          <div className="grid grid-cols-3 gap-2 bg-slate-950 p-3 rounded border border-slate-800 text-center font-mono text-xs mb-4">
            <div>
              <div className="text-[10px] text-slate-500">TEST MAE</div>
              <div className="font-bold text-slate-200">4,421 kWh</div>
            </div>
            <div>
              <div className="text-[10px] text-slate-500">sMAPE</div>
              <div className="font-bold text-slate-200">2.18%</div>
            </div>
            <div>
              <div className="text-[10px] text-slate-500">TEST R²</div>
              <div className="font-bold text-slate-200">0.9944</div>
            </div>
          </div>

          <div className="flex items-center justify-between text-xs font-mono">
            <span className="text-slate-500 flex items-center gap-1">
              <Info className="w-3.5 h-3.5" /> 4,416 sealed test hours
            </span>
            <Link to="/forecast" className="text-cyan-400 hover:text-cyan-300 font-bold flex items-center gap-1">
              Launch Forecaster <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};
