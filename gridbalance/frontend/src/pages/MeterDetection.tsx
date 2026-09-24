import React, { useState } from 'react';
import { ShieldAlert, Upload, CheckCircle2, AlertTriangle, FileText, Info, BarChart3, HelpCircle } from 'lucide-react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, BarChart, Bar } from 'recharts';
import { analyzeSingleMeter, type SingleMeterResult } from '../services/api';

export const MeterDetection: React.FC = () => {
  const [meterId, setMeterId] = useState('CONS_TEST_001');
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SingleMeterResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);
    try {
      if (file) {
        const text = await file.text();
        const lines = text.trim().split('\n');
        const headers = lines[0].split(',').map(h => h.trim());
        
        let mId = meterId;
        const readings: { date: string; consumption: number }[] = [];
        
        if (lines.length > 1) {
          for (let i = 1; i < lines.length; i++) {
            const parts = lines[i].split(',').map(p => p.trim());
            if (parts.length >= 2) {
              if (headers.includes('meter_id') || headers.includes('CONS_NO')) {
                mId = parts[0];
                readings.push({ date: parts[1], consumption: parseFloat(parts[2]) || 0.0 });
              } else {
                readings.push({ date: parts[0], consumption: parseFloat(parts[1]) || 0.0 });
              }
            }
          }
        }
        const res = await analyzeSingleMeter(mId, readings);
        setResult(res);
      } else {
        const res = await analyzeSingleMeter(meterId);
        setResult(res);
      }
    } catch (err: any) {
      setError(err.message || 'Analysis failed. Check your data format.');
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'Very High': return 'text-rose-400 bg-rose-500/10 border-rose-500/30';
      case 'High': return 'text-orange-400 bg-orange-500/10 border-orange-500/30';
      case 'Moderate': return 'text-amber-400 bg-amber-500/10 border-amber-500/30';
      default: return 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30';
    }
  };

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-3">
          <ShieldAlert className="w-6 h-6 text-teal-400" />
          Single Meter Tampering Detection
        </h1>
        <p className="text-xs sm:text-sm text-slate-400 mt-1">
          Analyze customer consumption time-series using the frozen 18-feature XGBoost classifier.
        </p>
      </div>

      {/* Upload / Input Form */}
      <div className="p-6 rounded-xl bg-slate-900 border border-slate-800 space-y-6">
        <h2 className="text-sm font-semibold text-white">Meter Input & Dataset Options</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-3">
            <label className="block text-xs font-medium text-slate-300">Meter Identifier (CONS_NO)</label>
            <input
              type="text"
              value={meterId}
              onChange={(e) => setMeterId(e.target.value)}
              className="w-full px-3 py-2 text-xs rounded-lg bg-slate-950 border border-slate-750 text-white focus:outline-none focus:border-teal-500"
              placeholder="e.g. CONS_42001"
            />

            <div className="pt-2">
              <label className="block text-xs font-medium text-slate-300 mb-2">Upload Customer Consumption CSV (Optional)</label>
              <label className="flex flex-col items-center justify-center p-4 border-2 border-dashed border-slate-750 hover:border-teal-500/50 rounded-lg cursor-pointer bg-slate-950/50 transition-colors">
                <Upload className="w-6 h-6 text-slate-400 mb-1" />
                <span className="text-xs text-slate-300 font-medium">
                  {file ? file.name : 'Click or drop CSV file'}
                </span>
                <span className="text-[10px] text-slate-500 mt-1">Supported: date, consumption</span>
                <input type="file" accept=".csv" onChange={handleFileUpload} className="hidden" />
              </label>
            </div>
          </div>

          <div className="p-4 rounded-lg bg-slate-950/80 border border-slate-800 space-y-3 text-xs">
            <span className="font-semibold text-teal-400 flex items-center gap-1.5">
              <FileText className="w-4 h-4" /> Supported CSV Format
            </span>
            <pre className="p-3 rounded bg-slate-900 text-slate-300 font-mono text-[11px] overflow-x-auto">
{`date,consumption
2014-01-01,14.5
2014-01-02,12.8
2014-01-03,0.0
2014-01-04,15.2`}
            </pre>
            <p className="text-[11px] text-slate-400">
              *If no file is uploaded, a benchmark customer profile will be generated automatically for evaluation.
            </p>
          </div>
        </div>

        {error && (
          <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <button
          onClick={handleAnalyze}
          disabled={loading}
          className="px-6 py-2.5 rounded-lg bg-teal-600 hover:bg-teal-500 disabled:opacity-50 text-white font-medium text-xs sm:text-sm flex items-center gap-2 transition-all shadow-lg shadow-teal-950"
        >
          {loading ? (
            <>
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              Extracting Features & Predicting...
            </>
          ) : (
            <>
              <ShieldAlert className="w-4 h-4" />
              Analyze Meter Profile
            </>
          )}
        </button>
      </div>

      {/* Analysis Results Display */}
      {result && (
        <div className="space-y-6">
          {/* Prediction Banner */}
          <div className="p-6 rounded-xl bg-slate-900 border border-slate-800 grid grid-cols-1 md:grid-cols-4 gap-6 items-center">
            <div className="space-y-1">
              <span className="text-xs text-slate-400 font-medium uppercase tracking-wider">Classification Result</span>
              <div className="flex items-center gap-2">
                {result.prediction === 'potential_tampering' ? (
                  <span className="text-lg font-bold text-rose-400 flex items-center gap-1.5">
                    <AlertTriangle className="w-5 h-5" /> Potential Tampering
                  </span>
                ) : (
                  <span className="text-lg font-bold text-emerald-400 flex items-center gap-1.5">
                    <CheckCircle2 className="w-5 h-5" /> Normal Pattern
                  </span>
                )}
              </div>
            </div>

            <div className="space-y-1">
              <span className="text-xs text-slate-400 font-medium uppercase tracking-wider">Model Tampering Score</span>
              <div className="text-2xl font-bold font-mono text-white">
                {(result.probability * 100).toFixed(1)}%
              </div>
              <p className="text-[10px] text-slate-500">Decision Threshold: {(result.threshold * 100).toFixed(0)}%</p>
            </div>

            <div className="space-y-1">
              <span className="text-xs text-slate-400 font-medium uppercase tracking-wider">Assessed Risk Level</span>
              <div>
                <span className={`inline-block px-3 py-1 rounded-full text-xs font-bold border ${getRiskColor(result.risk_level)}`}>
                  {result.risk_level} Risk
                </span>
              </div>
            </div>

            <div className="space-y-1 text-xs text-slate-400 border-l border-slate-800 pl-4">
              <span className="font-semibold text-slate-300">Meter ID:</span> {result.meter_id}
              <p className="text-[10px] text-slate-500 mt-1">Evaluated by 18-Feature XGBoost Engine</p>
            </div>
          </div>

          {/* Data Quality & SHAP Feature Contribution */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-4">
              <h3 className="text-xs font-semibold text-white uppercase tracking-wider flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-teal-400" />
                Data Quality Metrics
              </h3>
              <div className="space-y-3 text-xs">
                <div className="flex justify-between py-1.5 border-b border-slate-800 text-slate-300">
                  <span>Total Observations</span>
                  <span className="font-mono font-bold">{result.data_quality.observation_count} days</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-800 text-slate-300">
                  <span>Missing Values Count</span>
                  <span className="font-mono font-bold text-amber-400">{result.data_quality.missing_count}</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-800 text-slate-300">
                  <span>Missing Ratio</span>
                  <span className="font-mono font-bold">{(result.data_quality.missing_ratio * 100).toFixed(2)}%</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-800 text-slate-300">
                  <span>Longest Missing Streak</span>
                  <span className="font-mono font-bold">{result.data_quality.longest_missing_streak} days</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-800 text-slate-300">
                  <span>Missing Streaks Count</span>
                  <span className="font-mono font-bold">{result.data_quality.missing_streak_count}</span>
                </div>
              </div>
            </div>

            <div className="lg:col-span-2 p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-4">
              <h3 className="text-xs font-semibold text-white uppercase tracking-wider flex items-center gap-2">
                <HelpCircle className="w-4 h-4 text-teal-400" />
                Top Model Feature Contributions (SHAP)
              </h3>
              <div className="h-48 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={result.top_features} layout="vertical" margin={{ left: 40 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis type="number" stroke="#64748b" fontSize={10} />
                    <YAxis dataKey="feature" type="category" stroke="#64748b" fontSize={10} />
                    <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }} />
                    <Bar dataKey="importance" fill="#14b8a6" radius={[0, 4, 4, 0]} name="Feature Importance" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Consumption History Line Chart */}
          <div className="p-6 rounded-xl bg-slate-900 border border-slate-800 space-y-4">
            <h3 className="text-sm font-semibold text-white">Consumption History Trajectory</h3>
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

          {/* Research Disclaimer */}
          <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs flex items-start gap-3">
            <Info className="w-5 h-5 shrink-0 mt-0.5" />
            <div>
              <span className="font-semibold block mb-0.5">Research Classification Disclaimer</span>
              This prediction represents a statistical model score based on consumption time-series patterns. It does NOT independently prove physical meter tampering or illegal electricity theft.
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
