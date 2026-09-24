import React, { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ShieldAlert, ArrowLeft, Info, CheckCircle2, AlertTriangle, BarChart3, HelpCircle } from 'lucide-react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, BarChart, Bar } from 'recharts';
import { analyzeSingleMeter, type SingleMeterResult } from '../services/api';

export const MeterIntelligence: React.FC = () => {
  const { meterId } = useParams<{ meterId: string }>();
  const [result, setResult] = useState<SingleMeterResult | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    analyzeSingleMeter(meterId || 'CONS_001')
      .then((res) => {
        setResult(res);
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setLoading(false);
      });
  }, [meterId]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-24 space-y-4">
        <div className="w-8 h-8 border-4 border-teal-500 border-t-transparent rounded-full animate-spin"></div>
        <p className="text-xs text-slate-400">Loading Meter Intelligence Profile for {meterId}...</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="p-8 text-center bg-slate-900 rounded-xl border border-slate-800 space-y-4">
        <AlertTriangle className="w-8 h-8 text-rose-400 mx-auto" />
        <h2 className="text-sm font-semibold text-white">Meter Profile Not Found</h2>
        <Link to="/detection" className="text-xs text-teal-400 hover:underline">← Back to Detection</Link>
      </div>
    );
  }

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      {/* Navigation & Header */}
      <div className="space-y-3">
        <Link to="/detection/batch" className="inline-flex items-center gap-1.5 text-xs text-teal-400 hover:underline">
          <ArrowLeft className="w-3.5 h-3.5" /> Back to Batch Analysis
        </Link>
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold font-mono text-white flex items-center gap-3">
              <ShieldAlert className="w-6 h-6 text-teal-400" />
              Meter Intelligence: {result.meter_id}
            </h1>
            <p className="text-xs text-slate-400 mt-1">
              Deep-dive customer risk classification and data quality diagnostics.
            </p>
          </div>
          <div>
            <span className={`px-4 py-1.5 rounded-full text-xs font-bold border ${
              result.risk_level === 'Very High' ? 'text-rose-400 bg-rose-500/10 border-rose-500/30' :
              result.risk_level === 'High' ? 'text-orange-400 bg-orange-500/10 border-orange-500/30' :
              result.risk_level === 'Moderate' ? 'text-amber-400 bg-amber-500/10 border-amber-500/30' :
              'text-emerald-400 bg-emerald-500/10 border-emerald-500/30'
            }`}>
              {result.risk_level} Risk Profile
            </span>
          </div>
        </div>
      </div>

      {/* Main Score Banner */}
      <div className="p-6 rounded-xl bg-slate-900 border border-slate-800 grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="space-y-1">
          <span className="text-xs text-slate-400 uppercase tracking-wider">Classification Status</span>
          <div className="text-lg font-bold">
            {result.prediction === 'potential_tampering' ? (
              <span className="text-rose-400 flex items-center gap-2">
                <AlertTriangle className="w-5 h-5" /> Potential Tampering Flagged
              </span>
            ) : (
              <span className="text-emerald-400 flex items-center gap-2">
                <CheckCircle2 className="w-5 h-5" /> Normal Consumption Pattern
              </span>
            )}
          </div>
        </div>

        <div className="space-y-1">
          <span className="text-xs text-slate-400 uppercase tracking-wider">Model Probability Score</span>
          <div className="text-3xl font-bold font-mono text-white">
            {(result.probability * 100).toFixed(1)}%
          </div>
          <p className="text-[10px] text-slate-500">Decision Threshold: {(result.threshold * 100).toFixed(0)}%</p>
        </div>

        <div className="space-y-1 text-xs text-slate-400">
          <span className="text-slate-300 font-semibold block mb-1">Engine Information</span>
          <p>Frozen 18-Feature XGBoost Classifier</p>
          <p className="text-[10px] text-slate-500 mt-1">Research Test F1 = 0.4014 | PR-AUC = 0.4049</p>
        </div>
      </div>

      {/* Data Quality & SHAP Features */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-3 text-xs">
          <h3 className="font-semibold text-white uppercase tracking-wider flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-teal-400" /> Quality Diagnostics
          </h3>
          <div className="space-y-2 text-slate-300">
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span>Observation Count</span>
              <span className="font-mono font-bold">{result.data_quality.observation_count} days</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span>Missing Values</span>
              <span className="font-mono font-bold text-amber-400">{result.data_quality.missing_count}</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span>Missing Ratio</span>
              <span className="font-mono font-bold">{(result.data_quality.missing_ratio * 100).toFixed(2)}%</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span>Longest Missing Streak</span>
              <span className="font-mono font-bold">{result.data_quality.longest_missing_streak} days</span>
            </div>
          </div>
        </div>

        <div className="lg:col-span-2 p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-3">
          <h3 className="text-xs font-semibold text-white uppercase tracking-wider flex items-center gap-2">
            <HelpCircle className="w-4 h-4 text-teal-400" /> SHAP Feature Importance
          </h3>
          <div className="h-44 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={result.top_features} layout="vertical" margin={{ left: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis type="number" stroke="#64748b" fontSize={10} />
                <YAxis dataKey="feature" type="category" stroke="#64748b" fontSize={10} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }} />
                <Bar dataKey="importance" fill="#14b8a6" radius={[0, 4, 4, 0]} name="Feature Contribution" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Consumption Chart */}
      <div className="p-6 rounded-xl bg-slate-900 border border-slate-800 space-y-4">
        <h3 className="text-sm font-semibold text-white">Full Customer Load Trajectory</h3>
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={result.consumption_history}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="date" stroke="#64748b" fontSize={10} />
              <YAxis stroke="#64748b" fontSize={10} />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }} />
              <Line type="monotone" dataKey="consumption" stroke="#38bdf8" dot={false} strokeWidth={1.5} name="Consumption (kWh)" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs flex items-start gap-3">
        <Info className="w-5 h-5 shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold block mb-0.5">Formal Research Disclaimer</span>
          This classification represents a statistical model score based on historical consumption patterns. It does NOT independently establish that electricity theft or physical meter tampering occurred.
        </div>
      </div>
    </div>
  );
};
