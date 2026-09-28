import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { AlertTriangle, ShieldAlert, BarChart2, Layers, Search, RefreshCw, Info } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, BarChart, Bar, Cell } from 'recharts';
import { api } from '../services/api';
import type { SingleMeterResponse } from '../services/api';

export const MeterDetection: React.FC = () => {
  const [meterId, setMeterId] = useState<string>('CONS_SAMPLE_402');
  const [datesText, setDatesText] = useState<string>('2014-01-01, 2014-01-02, 2014-01-03, 2014-01-04, 2014-01-05, 2014-01-06, 2014-01-07, 2014-01-08, 2014-01-09, 2014-01-10, 2014-01-11, 2014-01-12, 2014-01-13, 2014-01-14');
  const [consumptionText, setConsumptionText] = useState<string>('14.2, 13.8, 15.0, 0.0, 0.0, 0.0, 1.1, 14.5, 15.1, 13.9, 14.0, 12.8, 13.5, 14.1');
  const [sampleType, setSampleType] = useState<'normal' | 'tampering' | 'custom'>('tampering');

  const [loading, setLoading] = useState<boolean>(false);
  const [step, setStep] = useState<'idle' | 'validating' | 'extracting' | 'predicting' | 'complete'>('idle');
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SingleMeterResponse | null>(null);

  const handleSampleLoad = (type: 'normal' | 'tampering') => {
    setSampleType(type);
    if (type === 'normal') {
      setMeterId('CONS_NORM_088');
      setDatesText('2014-01-01, 2014-01-02, 2014-01-03, 2014-01-04, 2014-01-05, 2014-01-06, 2014-01-07, 2014-01-08, 2014-01-09, 2014-01-10, 2014-01-11, 2014-01-12, 2014-01-13, 2014-01-14');
      setConsumptionText('12.5, 13.0, 12.8, 13.2, 12.9, 13.5, 14.0, 13.1, 12.7, 13.4, 13.0, 12.8, 13.2, 12.6');
    } else {
      setMeterId('CONS_TAMPER_109');
      setDatesText('2014-01-01, 2014-01-02, 2014-01-03, 2014-01-04, 2014-01-05, 2014-01-06, 2014-01-07, 2014-01-08, 2014-01-09, 2014-01-10, 2014-01-11, 2014-01-12, 2014-01-13, 2014-01-14');
      setConsumptionText('16.5, 17.0, 15.8, 0.0, 0.0, 0.0, 0.0, 0.0, 1.2, 16.0, 16.5, 0.0, 0.0, 15.9');
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (evt) => {
      const text = evt.target?.result as string;
      if (!text) return;
      const lines = text.split(/\r?\n/).filter(line => line.trim().length > 0);
      if (lines.length < 2) {
        setError("Uploaded CSV must contain a header and at least one data row.");
        return;
      }

      const headers = lines[0].split(',').map(s => s.trim());
      const firstRow = lines[1].split(',').map(s => s.trim());

      const mId = firstRow[0] || 'CONS_UPLOADED';
      const dates = headers.slice(1);
      const cons = firstRow.slice(1).map(v => parseFloat(v) || 0.0);

      setMeterId(mId);
      setDatesText(dates.join(', '));
      setConsumptionText(cons.join(', '));
      setSampleType('custom');
    };
    reader.readAsText(file);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    setStep('validating');
    const dates = datesText.split(',').map(s => s.trim()).filter(Boolean);
    const consumption = consumptionText.split(',').map(s => parseFloat(s.trim())).filter(v => !isNaN(v));

    if (dates.length === 0 || consumption.length === 0) {
      setError("Dates and consumption arrays cannot be empty.");
      setLoading(false);
      setStep('idle');
      return;
    }

    if (dates.length !== consumption.length) {
      setError(`Array length mismatch: ${dates.length} dates vs ${consumption.length} consumption values.`);
      setLoading(false);
      setStep('idle');
      return;
    }

    try {
      setStep('extracting');
      await new Promise(r => setTimeout(r, 200));

      setStep('predicting');
      const res = await api.analyzeMeter({
        meter_id: meterId || 'CONS_UNKNOWN',
        dates,
        consumption
      });

      setResult(res);

      const prevAnalyzed = parseInt(sessionStorage.getItem('gridbalance_analyzed_count') || '0', 10);
      const prevFlagged = parseInt(sessionStorage.getItem('gridbalance_flagged_count') || '0', 10);
      sessionStorage.setItem('gridbalance_analyzed_count', (prevAnalyzed + 1).toString());
      if (res.prediction === 'potential_tampering') {
        sessionStorage.setItem('gridbalance_flagged_count', (prevFlagged + 1).toString());
      }

      const historyStr = sessionStorage.getItem('gridbalance_meter_history') || '{}';
      const historyMap = JSON.parse(historyStr);
      historyMap[res.meter_id] = res;
      sessionStorage.setItem('gridbalance_meter_history', JSON.stringify(historyMap));

      setStep('complete');
    } catch (err: any) {
      setError(err.message || "Failed to execute meter analysis.");
      setStep('idle');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl sm:text-2xl font-bold text-slate-100 font-mono tracking-tight flex items-center gap-2.5">
              <ShieldAlert className="w-6 h-6 text-emerald-400" />
              METER TAMPERING DETECTION
            </h1>
            <p className="text-slate-400 text-xs mt-1">
              Analyze historical electricity consumption patterns using the frozen SGCC 18-feature classifier.
            </p>
          </div>

          <Link
            to="/detection/batch"
            className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-300 hover:text-emerald-400 hover:border-slate-700 text-xs font-mono transition-colors"
          >
            <Layers className="w-4 h-4" />
            Batch CSV Audit
          </Link>
        </div>

        {/* Phase 6 Status Strip */}
        <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono text-xs bg-slate-900 border border-slate-800 p-3 rounded-lg">
          <div>
            <span className="text-slate-500">Model:</span> <span className="text-slate-200 font-semibold">SGCC XGBoost</span>
          </div>
          <div>
            <span className="text-slate-500">Features:</span> <span className="text-slate-200 font-semibold">18 Features</span>
          </div>
          <div>
            <span className="text-slate-500">Threshold:</span> <span className="text-amber-400 font-bold">0.50</span>
          </div>
          <div>
            <span className="text-slate-500">Task:</span> <span className="text-emerald-400 font-semibold">Potential Tampering Class.</span>
          </div>
        </div>
      </div>

      {/* Form / Input Card */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
          <h2 className="text-xs font-mono uppercase tracking-wider text-slate-300 flex items-center gap-2">
            <Search className="w-4 h-4 text-emerald-400" />
            Input Consumption Series
          </h2>

          <div className="flex items-center gap-1.5 bg-slate-950 p-1 rounded-lg border border-slate-800">
            <button
              type="button"
              onClick={() => handleSampleLoad('normal')}
              className={`px-3 py-1 rounded text-xs font-mono transition-colors cursor-pointer ${
                sampleType === 'normal'
                  ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-bold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Normal Pattern
            </button>
            <button
              type="button"
              onClick={() => handleSampleLoad('tampering')}
              className={`px-3 py-1 rounded text-xs font-mono transition-colors cursor-pointer ${
                sampleType === 'tampering'
                  ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 font-bold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Tampering Pattern
            </button>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-xs font-mono text-slate-400 mb-1">Meter / Customer ID</label>
              <input
                type="text"
                value={meterId}
                onChange={(e) => { setMeterId(e.target.value); setSampleType('custom'); }}
                placeholder="e.g. CONS_NO_402"
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3.5 py-2 text-xs text-slate-100 font-mono focus:outline-none focus:border-emerald-500/50"
                required
              />
            </div>

            <div className="md:col-span-2">
              <label className="block text-xs font-mono text-slate-400 mb-1 flex items-center justify-between">
                <span>CSV File Quick Loader</span>
                <span className="text-slate-500 text-[11px]">Format: meter_id, date1, date2...</span>
              </label>
              <input
                type="file"
                accept=".csv, .txt"
                onChange={handleFileUpload}
                className="w-full text-xs text-slate-400 file:mr-3 file:py-1.5 file:px-3 file:rounded-md file:border-0 file:text-xs file:font-mono file:bg-slate-800 file:text-emerald-400 cursor-pointer"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-mono text-slate-400 mb-1">Dates Array (Comma-Separated)</label>
            <textarea
              rows={2}
              value={datesText}
              onChange={(e) => { setDatesText(e.target.value); setSampleType('custom'); }}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs text-slate-200 font-mono focus:outline-none focus:border-emerald-500/50"
              placeholder="2014-01-01, 2014-01-02, 2014-01-03..."
              required
            />
          </div>

          <div>
            <label className="block text-xs font-mono text-slate-400 mb-1">Daily Consumption kWh Array (Comma-Separated)</label>
            <textarea
              rows={2}
              value={consumptionText}
              onChange={(e) => { setConsumptionText(e.target.value); setSampleType('custom'); }}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs text-slate-200 font-mono focus:outline-none focus:border-emerald-500/50"
              placeholder="14.2, 13.8, 15.0, 0.0, 0.0..."
              required
            />
          </div>

          {error && (
            <div className="p-3 bg-rose-950/40 border border-rose-800/60 rounded-lg text-rose-300 text-xs font-mono flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {loading && (
            <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 space-y-2 font-mono text-xs">
              <div className="flex items-center justify-between text-slate-300">
                <span className="flex items-center gap-2">
                  <RefreshCw className="w-3.5 h-3.5 animate-spin text-emerald-400" />
                  Executing SGCC Pipeline...
                </span>
                <span className="text-slate-500 uppercase">{step}</span>
              </div>
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full sm:w-auto px-5 py-2.5 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold font-mono text-xs tracking-wider uppercase transition-colors shadow-md disabled:opacity-50 cursor-pointer flex items-center justify-center gap-2"
          >
            <ShieldAlert className="w-4 h-4" />
            {loading ? 'Processing Pipeline...' : 'Run SGCC Model Inference'}
          </button>
        </form>
      </div>

      {/* Results Panel */}
      {result && (
        <div className="space-y-6">
          <div className={`border rounded-xl p-6 ${
            result.prediction === 'potential_tampering'
              ? 'bg-rose-950/20 border-rose-800/80'
              : 'bg-emerald-950/20 border-emerald-800/80'
          }`}>
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <div className="text-xs font-mono uppercase tracking-widest text-slate-400">ANALYSIS RESULT</div>
                <div className="text-2xl font-extrabold font-mono mt-1 flex items-center gap-3">
                  <span className={result.prediction === 'potential_tampering' ? 'text-rose-400' : 'text-emerald-400'}>
                    {result.prediction === 'potential_tampering' ? 'POTENTIAL TAMPERING' : 'NORMAL CONSUMPTION'}
                  </span>
                  <span className={`text-xs px-2.5 py-1 rounded font-mono border ${
                    result.risk_level === 'High' ? 'bg-rose-500/20 border-rose-500/40 text-rose-300' :
                    result.risk_level === 'Medium' ? 'bg-amber-500/20 border-amber-500/40 text-amber-300' :
                    'bg-emerald-500/20 border-emerald-500/40 text-emerald-300'
                  }`}>
                    {result.risk_level.toUpperCase()} RISK
                  </span>
                </div>
              </div>

              <div className="flex items-center gap-4 font-mono text-center">
                <div className="bg-slate-900/90 px-4 py-2 rounded-lg border border-slate-800">
                  <div className="text-[10px] text-slate-400 uppercase">Tampering Probability</div>
                  <div className="text-lg font-bold text-slate-100">{(result.probability * 100).toFixed(1)}%</div>
                </div>
                <div className="bg-slate-900/90 px-4 py-2 rounded-lg border border-slate-800">
                  <div className="text-[10px] text-slate-400 uppercase">Decision Threshold</div>
                  <div className="text-lg font-bold text-slate-100">{(result.threshold * 100).toFixed(0)}%</div>
                </div>
              </div>
            </div>

            <div className="mt-4 p-3 rounded-lg bg-slate-950/60 border border-slate-800 text-xs text-slate-300 flex items-start gap-2.5">
              <Info className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
              <span>
                <strong>Research Evaluation Disclaimer:</strong> This result represents a machine-learning classification based on the supplied consumption pattern. It does not independently establish that electricity theft occurred.
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
              <h3 className="text-xs font-mono uppercase tracking-wider text-slate-300 flex items-center gap-2">
                <BarChart2 className="w-4 h-4 text-emerald-400" />
                Data Quality Signals
              </h3>

              <div className="space-y-3 font-mono text-xs">
                <div>
                  <div className="flex justify-between text-slate-400 mb-1">
                    <span>Observation Completeness</span>
                    <span className="text-slate-200">
                      {((1 - result.data_quality.missing_ratio) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-slate-950 h-2 rounded-full overflow-hidden border border-slate-800">
                    <div
                      className="bg-emerald-400 h-full transition-all duration-500"
                      style={{ width: `${(1 - result.data_quality.missing_ratio) * 100}%` }}
                    ></div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3 pt-2">
                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                    <div className="text-[10px] text-slate-500 uppercase">Total Observations</div>
                    <div className="text-base font-bold text-slate-200">{result.data_quality.observation_count} days</div>
                  </div>
                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                    <div className="text-[10px] text-slate-500 uppercase">Missing Values</div>
                    <div className="text-base font-bold text-slate-200">{result.data_quality.missing_count} days</div>
                  </div>
                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                    <div className="text-[10px] text-slate-500 uppercase">Longest Streak</div>
                    <div className="text-base font-bold text-slate-200">{result.data_quality.longest_missing_streak} days</div>
                  </div>
                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                    <div className="text-[10px] text-slate-500 uppercase">Streak Count</div>
                    <div className="text-base font-bold text-slate-200">{result.data_quality.missing_streak_count} gaps</div>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
              <h3 className="text-xs font-mono uppercase tracking-wider text-slate-300 flex items-center justify-between">
                <span className="flex items-center gap-2">
                  <Layers className="w-4 h-4 text-emerald-400" />
                  Top Model Signals
                </span>
                <span className="text-[11px] text-slate-500 font-normal">SHAP-Aligned Weights</span>
              </h3>

              {result.top_features && result.top_features.length > 0 ? (
                <div className="h-52 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      layout="vertical"
                      data={result.top_features}
                      margin={{ top: 5, right: 30, left: 80, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                      <XAxis type="number" stroke="#64748b" fontSize={10} tickFormatter={(v) => v.toFixed(2)} />
                      <YAxis type="category" dataKey="feature" stroke="#94a3b8" fontSize={10} tickLine={false} width={110} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#090d16', borderColor: '#334155', fontSize: '11px', fontFamily: 'monospace' }}
                        formatter={(val: any) => [typeof val === 'number' ? val.toFixed(4) : val, 'Weight']}
                      />
                      <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                        {result.top_features.map((_, index) => (
                          <Cell key={`cell-${index}`} fill={index === 0 ? '#34d399' : '#0284c7'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <p className="text-xs text-slate-500 italic font-mono py-8 text-center">
                  Feature explanation unavailable for this prediction.
                </p>
              )}
            </div>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
            <h3 className="text-xs font-mono uppercase tracking-wider text-slate-300 flex items-center justify-between">
              <span className="flex items-center gap-2">
                <BarChart2 className="w-4 h-4 text-emerald-400" />
                Uploaded Consumption Trajectory (kWh)
              </span>
              <span className="text-xs text-slate-400 font-mono">{result.consumption_history.length} Observations</span>
            </h3>

            <div className="h-60 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={result.consumption_history} margin={{ top: 10, right: 20, left: 10, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="date" stroke="#64748b" fontSize={10} tickLine={false} />
                  <YAxis stroke="#64748b" fontSize={10} tickFormatter={(v) => `${v} kWh`} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#090d16', borderColor: '#334155', fontSize: '12px', fontFamily: 'monospace' }}
                    formatter={(val: any) => [`${val} kWh`, 'Consumption']}
                  />
                  <Line
                    type="monotone"
                    dataKey="consumption"
                    stroke={result.prediction === 'potential_tampering' ? '#f87171' : '#34d399'}
                    strokeWidth={2}
                    dot={{ r: 3, fill: result.prediction === 'potential_tampering' ? '#f87171' : '#34d399' }}
                    activeDot={{ r: 5 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
