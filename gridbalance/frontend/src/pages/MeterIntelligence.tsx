import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, BarChart2, Layers, Info } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, BarChart, Bar } from 'recharts';
import type { SingleMeterResponse } from '../services/api';

export const MeterIntelligence: React.FC = () => {
  const { meterId } = useParams<{ meterId: string }>();
  const [meterData, setMeterData] = useState<SingleMeterResponse | null>(null);

  useEffect(() => {
    if (!meterId) return;
    const historyStr = sessionStorage.getItem('gridbalance_meter_history') || '{}';
    try {
      const historyMap = JSON.parse(historyStr);
      if (historyMap[meterId]) {
        setMeterData(historyMap[meterId]);
      }
    } catch {
      // Ignore
    }
  }, [meterId]);

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Link to="/detection/batch" className="p-2 rounded bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors">
          <ArrowLeft className="w-4 h-4" />
        </Link>
        <div>
          <h1 className="text-2xl font-bold font-mono text-slate-100 flex items-center gap-2">
            METER INTELLIGENCE: <span className="text-emerald-400">{meterId}</span>
          </h1>
          <p className="text-xs text-slate-400 font-mono">Individual Customer Consumption Profile & Risk Analysis</p>
        </div>
      </div>

      {meterData ? (
        <div className="space-y-6">
          {/* Assessment Card */}
          <div className={`border rounded-xl p-6 ${
            meterData.prediction === 'potential_tampering' ? 'bg-rose-950/20 border-rose-800' : 'bg-emerald-950/20 border-emerald-800'
          }`}>
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <div className="text-xs font-mono uppercase text-slate-400">Classification Assessment</div>
                <div className="text-2xl font-bold font-mono mt-1 flex items-center gap-3">
                  <span className={meterData.prediction === 'potential_tampering' ? 'text-rose-400' : 'text-emerald-400'}>
                    {meterData.prediction === 'potential_tampering' ? 'POTENTIAL TAMPERING' : 'NORMAL'}
                  </span>
                  <span className="text-xs px-2.5 py-1 rounded font-mono bg-slate-900 border border-slate-700 text-slate-300">
                    Risk Level: {meterData.risk_level}
                  </span>
                </div>
              </div>

              <div className="flex items-center gap-4 font-mono text-center">
                <div className="bg-slate-900 px-3.5 py-2 rounded border border-slate-800">
                  <div className="text-[10px] text-slate-500 uppercase">Probability</div>
                  <div className="text-lg font-bold text-slate-100">{(meterData.probability * 100).toFixed(1)}%</div>
                </div>
                <div className="bg-slate-900 px-3.5 py-2 rounded border border-slate-800">
                  <div className="text-[10px] text-slate-500 uppercase">Threshold</div>
                  <div className="text-lg font-bold text-slate-100">50%</div>
                </div>
              </div>
            </div>
          </div>

          {/* Grid Layout */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Data Quality */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-3 font-mono text-xs">
              <h3 className="text-sm font-bold text-slate-200 flex items-center gap-2">
                <BarChart2 className="w-4 h-4 text-emerald-400" /> Data Quality Signals
              </h3>
              <div className="grid grid-cols-2 gap-3 pt-2">
                <div className="bg-slate-950 p-3 rounded border border-slate-800">
                  <div className="text-[10px] text-slate-500">OBSERVATIONS</div>
                  <div className="text-base font-bold text-slate-200">{meterData.data_quality.observation_count} days</div>
                </div>
                <div className="bg-slate-950 p-3 rounded border border-slate-800">
                  <div className="text-[10px] text-slate-500">MISSING VALUES</div>
                  <div className="text-base font-bold text-slate-200">{meterData.data_quality.missing_count} days</div>
                </div>
                <div className="bg-slate-950 p-3 rounded border border-slate-800">
                  <div className="text-[10px] text-slate-500">LONGEST GAP</div>
                  <div className="text-base font-bold text-slate-200">{meterData.data_quality.longest_missing_streak} days</div>
                </div>
                <div className="bg-slate-950 p-3 rounded border border-slate-800">
                  <div className="text-[10px] text-slate-500">MISSING STREAKS</div>
                  <div className="text-base font-bold text-slate-200">{meterData.data_quality.missing_streak_count} gaps</div>
                </div>
              </div>
            </div>

            {/* Top Features */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-3">
              <h3 className="text-sm font-mono font-bold text-slate-200 flex items-center gap-2">
                <Layers className="w-4 h-4 text-emerald-400" /> Feature Importances
              </h3>
              <div className="h-44 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart layout="vertical" data={meterData.top_features} margin={{ top: 5, right: 20, left: 70, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                    <XAxis type="number" stroke="#64748b" fontSize={10} />
                    <YAxis type="category" dataKey="feature" stroke="#94a3b8" fontSize={10} width={90} />
                    <Tooltip contentStyle={{ backgroundColor: '#090d16', borderColor: '#334155', fontSize: '11px', fontFamily: 'monospace' }} />
                    <Bar dataKey="importance" fill="#34d399" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Consumption History Line Chart */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-3">
            <h3 className="text-sm font-mono font-bold text-slate-200">Consumption Curve</h3>
            <div className="h-60 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={meterData.consumption_history}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="date" stroke="#64748b" fontSize={10} />
                  <YAxis stroke="#64748b" fontSize={10} />
                  <Tooltip contentStyle={{ backgroundColor: '#090d16', borderColor: '#334155', fontSize: '11px', fontFamily: 'monospace' }} />
                  <Line type="monotone" dataKey="consumption" stroke="#34d399" strokeWidth={2} dot={{ r: 2 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      ) : (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-12 text-center font-mono space-y-4">
          <Info className="w-10 h-10 text-slate-500 mx-auto" />
          <p className="text-slate-300 text-sm">No analysis history found in session for meter ID: <span className="text-emerald-400">{meterId}</span></p>
          <p className="text-slate-500 text-xs max-w-md mx-auto">
            Execute a single meter analysis or upload a batch CSV dataset to populate profile intelligence.
          </p>
          <Link to="/detection" className="inline-block px-4 py-2 rounded bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold text-xs uppercase tracking-wider">
            Go to Meter Detection
          </Link>
        </div>
      )}
    </div>
  );
};
