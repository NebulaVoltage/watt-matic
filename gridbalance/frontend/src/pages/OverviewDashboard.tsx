import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { ShieldAlert, LineChart, ArrowRight, Zap } from 'lucide-react';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid, PieChart, Pie, Cell } from 'recharts';
import { fetchDashboardSummary } from '../services/api';

export const OverviewDashboard: React.FC = () => {
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    fetchDashboardSummary()
      .then((res) => {
        setData(res);
      })
      .catch((err) => {
        console.error(err);
      });
  }, []);

  const sampleChartData = [
    { time: '00:00', load: 112000, forecast: 114000 },
    { time: '04:00', load: 98000, forecast: 99500 },
    { time: '08:00', load: 135000, forecast: 134200 },
    { time: '12:00', load: 148000, forecast: 149100 },
    { time: '16:00', load: 152000, forecast: 151000 },
    { time: '20:00', load: 165000, forecast: 166400 },
    { time: '23:00', load: 128450, forecast: 130100 },
  ];

  const pieData = [
    { name: 'Normal Meters', value: data?.recent_metrics?.normal_meters || 38755, color: '#10b981' },
    { name: 'Potential Tampering', value: data?.recent_metrics?.potential_tampering_flagged || 3612, color: '#f43f5e' },
  ];

  return (
    <div className="space-y-8">
      {/* Hero Header */}
      <section className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-slate-900 via-slate-900 to-teal-950/40 p-8 border border-slate-800 shadow-2xl">
        <div className="absolute -right-12 -top-12 w-64 h-64 bg-teal-500/10 rounded-full blur-3xl pointer-events-none"></div>
        <div className="max-w-3xl space-y-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-teal-500/10 border border-teal-500/30 text-teal-400 text-xs font-mono">
            <Zap className="w-3.5 h-3.5" />
            GRIDBALANCE PLATFORM v1.0
          </div>
          <h1 className="text-4xl font-extrabold tracking-tight text-white sm:text-5xl">
            AI-Powered Smart Grid Intelligence
          </h1>
          <p className="text-slate-300 text-sm sm:text-base leading-relaxed">
            Detect anomalous meter behavior and forecast electricity demand using research-validated machine learning algorithms.
          </p>

          <div className="pt-4 flex flex-wrap gap-4">
            <Link
              to="/detection"
              className="inline-flex items-center gap-2 px-5 py-3 rounded-lg bg-teal-600 hover:bg-teal-500 text-white font-medium text-xs sm:text-sm transition-all shadow-lg shadow-teal-900/30"
            >
              <ShieldAlert className="w-4 h-4" />
              Analyze Meter
              <ArrowRight className="w-4 h-4 ml-1" />
            </Link>

            <Link
              to="/forecast"
              className="inline-flex items-center gap-2 px-5 py-3 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 font-medium text-xs sm:text-sm transition-all"
            >
              <LineChart className="w-4 h-4 text-teal-400" />
              Forecast Load
            </Link>
          </div>
        </div>
      </section>

      {/* Model Status Banner */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-emerald-500/10 border border-emerald-500/30 rounded-lg">
              <ShieldAlert className="w-5 h-5 text-emerald-400" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold text-white">SGCC Meter Tampering Engine</span>
                <span className="inline-flex items-center gap-1 text-[11px] font-medium text-emerald-400">
                  <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span> Online
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">18-Feature XGBoost Classifier • Fixed Threshold 0.50</p>
            </div>
          </div>
          <div className="text-right text-xs">
            <span className="text-slate-400">Test F1:</span> <span className="font-mono text-emerald-400 font-bold">0.4014</span>
          </div>
        </div>

        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-teal-500/10 border border-teal-500/30 rounded-lg">
              <LineChart className="w-5 h-5 text-teal-400" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold text-white">UCI Load Forecasting Engine</span>
                <span className="inline-flex items-center gap-1 text-[11px] font-medium text-emerald-400">
                  <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span> Online
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">29-Feature XGBoost Regressor • Next-Hour Horizon</p>
            </div>
          </div>
          <div className="text-right text-xs">
            <span className="text-slate-400">Test R²:</span> <span className="font-mono text-teal-400 font-bold">0.9944</span>
          </div>
        </div>
      </div>

      {/* Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
          <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">Meters Analyzed</span>
          <div className="text-3xl font-bold font-mono text-white">
            {data?.recent_metrics?.total_meters_analyzed?.toLocaleString() || '42,367'}
          </div>
          <p className="text-[11px] text-slate-500">Benchmark SGCC Customer Profiles</p>
        </div>

        <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
          <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">Potential Tampering</span>
          <div className="text-3xl font-bold font-mono text-rose-400">
            {data?.recent_metrics?.potential_tampering_flagged?.toLocaleString() || '3,612'}
          </div>
          <p className="text-[11px] text-slate-500">Flagged Anomalous Load Profiles (8.5%)</p>
        </div>

        <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
          <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">Normal Meters</span>
          <div className="text-3xl font-bold font-mono text-emerald-400">
            {data?.recent_metrics?.normal_meters?.toLocaleString() || '38,755'}
          </div>
          <p className="text-[11px] text-slate-500">Standard Consumption Patterns</p>
        </div>

        <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
          <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">Next-Hour Forecast</span>
          <div className="text-3xl font-bold font-mono text-teal-400">
            {data?.recent_metrics?.next_hour_forecast_load_kwh?.toLocaleString() || '128,450.5'} <span className="text-xs text-slate-400 font-sans">kWh</span>
          </div>
          <p className="text-[11px] text-slate-500">Predicted System Load (Time t+1)</p>
        </div>
      </div>

      {/* System Analytics Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 p-6 rounded-xl bg-slate-900 border border-slate-800 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <LineChart className="w-4 h-4 text-teal-400" />
              Aggregate System Load Profile & Forecast Trend
            </h3>
            <span className="text-xs text-slate-400 font-mono">1-Hour Resolution</span>
          </div>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={sampleChartData}>
                <defs>
                  <linearGradient id="colorLoad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#14b8a6" stopOpacity={0.4}/>
                    <stop offset="95%" stopColor="#14b8a6" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="time" stroke="#64748b" fontSize={11} />
                <YAxis stroke="#64748b" fontSize={11} domain={['auto', 'auto']} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }} />
                <Area type="monotone" dataKey="load" stroke="#14b8a6" fillOpacity={1} fill="url(#colorLoad)" name="Load (kWh)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="p-6 rounded-xl bg-slate-900 border border-slate-800 space-y-4">
          <h3 className="text-sm font-semibold text-white flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-rose-400" />
            Detection Class Distribution
          </h3>
          <div className="h-52 w-full flex items-center justify-center">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={pieData} cx="50%" cy="50%" innerRadius={50} outerRadius={75} paddingAngle={4} dataKey="value">
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="space-y-2 text-xs">
            <div className="flex justify-between items-center text-slate-300">
              <span className="flex items-center gap-2"><span className="w-2.5 h-2.5 rounded-full bg-emerald-500"></span> Normal Meters</span>
              <span className="font-mono font-semibold">38,755 (91.5%)</span>
            </div>
            <div className="flex justify-between items-center text-slate-300">
              <span className="flex items-center gap-2"><span className="w-2.5 h-2.5 rounded-full bg-rose-500"></span> Potential Tampering</span>
              <span className="font-mono font-semibold text-rose-400">3,612 (8.5%)</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
