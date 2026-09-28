import React, { useState } from 'react';
import { Zap, Clock, AlertTriangle, RefreshCw, BarChart2 } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceDot } from 'recharts';
import { api } from '../services/api';
import type { ForecastResponse, LoadPointInput } from '../services/api';

export const LoadForecasting: React.FC = () => {
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ForecastResponse | null>(null);

  // Generate 200 sample hourly observations starting from 2014-06-20
  const generateDefaultHistory = (): LoadPointInput[] => {
    const points: LoadPointInput[] = [];
    const baseDt = new Date('2014-06-20T00:00:00');
    for (let i = 0; i < 200; i++) {
      const dt = new Date(baseDt.getTime() + i * 3600 * 1000);
      const hour = dt.getHours();
      const dailyPattern = 110000 + Math.sin((hour / 24) * 2 * Math.PI - Math.PI / 2) * 25000;
      const noise = (Math.random() - 0.5) * 2000;
      const formattedTs = dt.toISOString().replace('T', ' ').slice(0, 19);
      points.push({
        timestamp: formattedTs,
        load_kwh: Math.round(dailyPattern + noise)
      });
    }
    return points;
  };

  const [history, setHistory] = useState<LoadPointInput[]>(generateDefaultHistory());

  const handleGenerateSample = () => {
    const fresh = generateDefaultHistory();
    setHistory(fresh);
    setResult(null);
    setError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (history.length < 168) {
      setError(`Insufficient history: minimum 168 hours required to compute 168-hr lag & rolling features. (Currently provided: ${history.length} hrs).`);
      return;
    }

    setLoading(true);
    try {
      const res = await api.forecastNextHour(history);
      setResult(res);

      // Save last forecast to session storage for homepage telemetry
      sessionStorage.setItem('gridbalance_last_forecast', res.predicted_load_kwh.toLocaleString());
    } catch (err: any) {
      setError(err.message || "Failed to generate next-hour load forecast.");
    } finally {
      setLoading(false);
    }
  };

  const currentLoad = history[history.length - 1]?.load_kwh || 0;
  const predictedLoad = result?.predicted_load_kwh || 0;
  const loadDiff = result ? predictedLoad - currentLoad : 0;
  const pctChange = result && currentLoad > 0 ? (loadDiff / currentLoad) * 100 : 0;

  // Build chart dataset appending the predicted point at target_timestamp
  const chartData = [...history.map(h => ({ timestamp: h.timestamp.slice(5, 16), load_kwh: h.load_kwh, isPrediction: false }))];
  if (result) {
    chartData.push({
      timestamp: result.target_timestamp.slice(5, 16),
      load_kwh: result.predicted_load_kwh,
      isPrediction: true
    });
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl sm:text-3xl font-bold text-slate-100 font-mono tracking-tight flex items-center gap-3">
          <Zap className="w-7 h-7 text-cyan-400" />
          ELECTRICITY LOAD FORECASTING
        </h1>
        <p className="text-slate-400 text-sm mt-1">
          Predict next-hour ($t+1$) aggregate grid electricity demand using causal historical load features and frozen XGBoost regression.
        </p>
      </div>

      {/* Input & Requirement Notice */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 className="text-sm font-mono uppercase tracking-wider text-slate-300 flex items-center gap-2">
              <Clock className="w-4 h-4 text-cyan-400" />
              Historical Hourly Load Sequence Input
            </h2>
            <p className="text-xs text-slate-400 font-mono mt-1">
              Minimum history required: <span className="text-cyan-400 font-bold">168 hours (7 days)</span> for 168-hr lag & rolling window calculations.
            </p>
          </div>

          <button
            type="button"
            onClick={handleGenerateSample}
            className="px-3.5 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-cyan-300 border border-cyan-500/30 text-xs font-mono transition-colors cursor-pointer"
          >
            Refresh Sample Series ({history.length} hrs)
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="bg-slate-950 p-4 rounded-lg border border-slate-800 font-mono text-xs space-y-2">
            <div className="flex justify-between text-slate-400">
              <span>Sequence Range: {history[0]?.timestamp} to {history[history.length - 1]?.timestamp}</span>
              <span className="text-emerald-400">{history.length} Hourly Observations</span>
            </div>
            <div className="text-[11px] text-slate-500">
              Latest observation ($t$): <span className="text-slate-200 font-bold">{currentLoad.toLocaleString()} kWh</span> at {history[history.length - 1]?.timestamp}
            </div>
          </div>

          {error && (
            <div className="p-3 bg-rose-950/40 border border-rose-800/60 rounded text-xs font-mono text-rose-300 flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="px-6 py-2.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold font-mono text-xs tracking-wider uppercase transition-colors shadow-md disabled:opacity-50 cursor-pointer flex items-center gap-2"
          >
            {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
            {loading ? 'Running UCI Regressor...' : 'Forecast Next-Hour Load (t+1)'}
          </button>
        </form>
      </div>

      {/* Forecast Result Banner (Phase 12) */}
      {result && (
        <div className="space-y-6">
          <div className="bg-cyan-950/20 border border-cyan-800/80 rounded-xl p-6 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <div className="text-xs font-mono uppercase tracking-widest text-slate-400">NEXT-HOUR FORECAST (t+1)</div>
                <div className="text-3xl sm:text-4xl font-extrabold font-mono text-cyan-400 mt-1">
                  {result.predicted_load_kwh.toLocaleString()} kWh
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4 font-mono text-center">
                <div className="bg-slate-900/90 px-4 py-2 rounded-lg border border-slate-800">
                  <div className="text-[10px] text-slate-400 uppercase">Current Load (t)</div>
                  <div className="text-base font-bold text-slate-200">{currentLoad.toLocaleString()} kWh</div>
                </div>

                <div className="bg-slate-900/90 px-4 py-2 rounded-lg border border-slate-800">
                  <div className="text-[10px] text-slate-400 uppercase">Expected Change</div>
                  <div className={`text-base font-bold ${loadDiff >= 0 ? 'text-emerald-400' : 'text-amber-400'}`}>
                    {loadDiff >= 0 ? '+' : ''}{loadDiff.toFixed(1)} kWh ({pctChange >= 0 ? '+' : ''}{pctChange.toFixed(2)}%)
                  </div>
                </div>

                <div className="bg-slate-900/90 px-4 py-2 rounded-lg border border-slate-800">
                  <div className="text-[10px] text-slate-400 uppercase">Target Timestamp</div>
                  <div className="text-xs font-bold text-cyan-300 mt-1">{result.target_timestamp}</div>
                </div>
              </div>
            </div>

            {/* Model & Horizon Metadata */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 font-mono text-xs bg-slate-950/60 p-3 rounded-lg border border-slate-800/80">
              <div>
                <span className="text-slate-500">Model Engine:</span>{' '}
                <span className="text-slate-200 font-semibold">{result.model_info.model_name || 'XGBoost Regressor'}</span>
              </div>
              <div>
                <span className="text-slate-500">Features Extracted:</span>{' '}
                <span className="text-slate-200 font-semibold">{result.model_info.feature_count || 29} Causal Features</span>
              </div>
              <div>
                <span className="text-slate-500">Forecast Horizon:</span>{' '}
                <span className="text-cyan-400 font-bold">Strictly Single-Step (t+1 hr)</span>
              </div>
            </div>
          </div>

          {/* Interactive Historical Load & Forecast Curve */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
            <h3 className="text-sm font-mono uppercase tracking-wider text-slate-300 flex items-center justify-between">
              <span className="flex items-center gap-2">
                <BarChart2 className="w-4 h-4 text-cyan-400" />
                200-Hour Load Trajectory & Next-Hour Forecast Target
              </span>
              <span className="text-xs text-slate-400 font-mono">Cyan Dot = Predicted Target (t+1)</span>
            </h3>

            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 10, right: 30, left: 20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="timestamp" stroke="#64748b" fontSize={10} interval={23} />
                  <YAxis stroke="#64748b" fontSize={10} tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#090d16', borderColor: '#334155', fontSize: '12px', fontFamily: 'monospace' }}
                    formatter={(val: any, _name: any, item: any) => [
                      `${val.toLocaleString()} kWh`,
                      item.payload.isPrediction ? 'NEXT-HOUR FORECAST (t+1)' : 'Historical Load (t)'
                    ]}
                  />
                  <Line
                    type="monotone"
                    dataKey="load_kwh"
                    stroke="#0284c7"
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 4 }}
                  />
                  {result && (
                    <ReferenceDot
                      x={result.target_timestamp.slice(5, 16)}
                      y={result.predicted_load_kwh}
                      r={7}
                      fill="#22d3ee"
                      stroke="#090d16"
                      strokeWidth={2}
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>

            <p className="text-[11px] text-slate-500 italic font-mono">
              Note: The UCI forecasting engine is frozen for single-step next-hour predictions. Multi-step horizon projections are not claimed or extrapolated.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
