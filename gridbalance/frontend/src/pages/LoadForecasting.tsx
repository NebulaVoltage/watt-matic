import React, { useState } from 'react';
import { LineChart, Upload, ArrowUpRight, ArrowDownRight, Zap, FileText } from 'lucide-react';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid, BarChart, Bar } from 'recharts';
import { forecastNextHour, forecastFromCSV, type ForecastResult } from '../services/api';

export const LoadForecasting: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ForecastResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleForecast = async () => {
    setLoading(true);
    setError(null);
    try {
      if (file) {
        const res = await forecastFromCSV(file);
        setResult(res);
      } else {
        const res = await forecastNextHour();
        setResult(res);
      }
    } catch (err: any) {
      setError(err.message || 'Forecasting failed. Check your data format.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-3">
          <LineChart className="w-6 h-6 text-teal-400" />
          Next-Hour Electricity Load Forecasting
        </h1>
        <p className="text-xs sm:text-sm text-slate-400 mt-1">
          Predict aggregate system demand (Load at time t+1) using the frozen 29-feature XGBoost Regressor (Test R² = 0.9944).
        </p>
      </div>

      {/* Upload & Form */}
      <div className="p-6 rounded-xl bg-slate-900 border border-slate-800 space-y-6">
        <h2 className="text-sm font-semibold text-white">Historical Load Input Options</h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <label className="block text-xs font-medium text-slate-300">Upload Historical Load CSV (Optional)</label>
            <label className="flex flex-col items-center justify-center p-6 border-2 border-dashed border-slate-750 hover:border-teal-500/50 rounded-lg cursor-pointer bg-slate-950/50 transition-colors">
              <Upload className="w-6 h-6 text-teal-400 mb-1" />
              <span className="text-xs text-slate-300 font-medium">
                {file ? file.name : 'Click or drop historical load CSV file'}
              </span>
              <span className="text-[10px] text-slate-500 mt-1">Supported: timestamp, load_kwh</span>
              <input type="file" accept=".csv" onChange={handleFileUpload} className="hidden" />
            </label>
          </div>

          <div className="p-4 rounded-lg bg-slate-950/80 border border-slate-800 space-y-3 text-xs">
            <span className="font-semibold text-teal-400 flex items-center gap-1.5">
              <FileText className="w-4 h-4" /> Supported Format
            </span>
            <pre className="p-3 rounded bg-slate-900 text-slate-300 font-mono text-[11px] overflow-x-auto">
{`timestamp,load_kwh
2014-12-24 18:00:00,128500.4
2014-12-24 19:00:00,131200.0
2014-12-24 20:00:00,129800.5`}
            </pre>
            <p className="text-[11px] text-slate-400">
              *If no CSV is uploaded, a benchmark 168-hour historical load profile will be evaluated automatically.
            </p>
          </div>
        </div>

        {error && (
          <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-center gap-2">
            <Zap className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <button
          onClick={handleForecast}
          disabled={loading}
          className="px-6 py-2.5 rounded-lg bg-teal-600 hover:bg-teal-500 disabled:opacity-50 text-white font-medium text-xs sm:text-sm flex items-center gap-2 transition-all shadow-lg shadow-teal-950"
        >
          {loading ? (
            <>
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              Generating Forecast...
            </>
          ) : (
            <>
              <LineChart className="w-4 h-4" />
              Generate Next-Hour Load Forecast
            </>
          )}
        </button>
      </div>

      {/* Forecast Output */}
      {result && (
        <div className="space-y-6">
          {/* Key Metric Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-1">
              <span className="text-xs text-slate-400 uppercase tracking-wider">Current Load (Time t)</span>
              <div className="text-3xl font-bold font-mono text-white">
                {result.current_load_kwh.toLocaleString()} <span className="text-xs text-slate-400 font-sans">kWh</span>
              </div>
              <p className="text-[10px] text-slate-500">Most Recent Observed Demand</p>
            </div>

            <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-1">
              <span className="text-xs text-slate-400 uppercase tracking-wider">Next-Hour Forecast (Time t+1)</span>
              <div className="text-3xl font-bold font-mono text-teal-400">
                {result.predicted_load_kwh.toLocaleString()} <span className="text-xs text-slate-400 font-sans">kWh</span>
              </div>
              <p className="text-[10px] text-slate-500">Forecast Horizon: {result.forecast_horizon}</p>
            </div>

            <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-1">
              <span className="text-xs text-slate-400 uppercase tracking-wider">Change From Current</span>
              <div className={`text-3xl font-bold font-mono flex items-center gap-1 ${result.change_from_current_kwh >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                {result.change_from_current_kwh >= 0 ? <ArrowUpRight className="w-6 h-6" /> : <ArrowDownRight className="w-6 h-6" />}
                {Math.abs(result.change_from_current_kwh).toLocaleString()} <span className="text-xs font-sans">kWh ({result.change_percentage >= 0 ? '+' : ''}{result.change_percentage}%)</span>
              </div>
              <p className="text-[10px] text-slate-500">Model: {result.model}</p>
            </div>
          </div>

          {/* Forecast Chart */}
          <div className="p-6 rounded-xl bg-slate-900 border border-slate-800 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-white">Historical Load & Next-Hour Forecast Step</h3>
              <span className="text-xs text-slate-400 font-mono flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-teal-400"></span> Actual Load
                <span className="w-2.5 h-2.5 rounded-full bg-rose-400"></span> Next-Hour Forecast
              </span>
            </div>
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={result.historical_chart}>
                  <defs>
                    <linearGradient id="colorForecast" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#14b8a6" stopOpacity={0.4}/>
                      <stop offset="95%" stopColor="#14b8a6" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="timestamp" stroke="#64748b" fontSize={10} />
                  <YAxis stroke="#64748b" fontSize={10} domain={['auto', 'auto']} />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }} />
                  <Area type="monotone" dataKey="load_kwh" stroke="#14b8a6" fillOpacity={1} fill="url(#colorForecast)" name="Load (kWh)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Feature Importance */}
          <div className="p-6 rounded-xl bg-slate-900 border border-slate-800 space-y-4">
            <h3 className="text-xs font-semibold text-white uppercase tracking-wider">Top Driving Features for Forecast Step</h3>
            <div className="h-40 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={result.top_features} layout="vertical" margin={{ left: 40 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis type="number" stroke="#64748b" fontSize={10} />
                  <YAxis dataKey="feature" type="category" stroke="#64748b" fontSize={10} />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }} />
                  <Bar dataKey="importance" fill="#0d9488" radius={[0, 4, 4, 0]} name="Feature Importance" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
