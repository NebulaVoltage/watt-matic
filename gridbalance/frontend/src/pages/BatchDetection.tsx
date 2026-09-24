import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Layers, Upload, Download, Search, Filter, AlertTriangle, CheckCircle2, ChevronLeft, ChevronRight } from 'lucide-react';
import { analyzeBatchMeters, type BatchMeterResult } from '../services/api';

export const BatchDetection: React.FC = () => {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<BatchMeterResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Search & Filter state
  const [search, setSearch] = useState('');
  const [filterRisk, setFilterRisk] = useState<string>('all');
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleBatchProcess = async () => {
    if (!file) {
      setError('Please select a batch CSV file first.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await analyzeBatchMeters(file);
      setData(res);
      setCurrentPage(1);
    } catch (err: any) {
      setError(err.message || 'Batch processing failed.');
    } finally {
      setLoading(false);
    }
  };

  const filteredResults = data?.results.filter((item) => {
    const matchesSearch = item.meter_id.toLowerCase().includes(search.toLowerCase());
    const matchesRisk = filterRisk === 'all' || item.risk_level.toLowerCase() === filterRisk.toLowerCase();
    return matchesSearch && matchesRisk;
  }) || [];

  const totalPages = Math.ceil(filteredResults.length / itemsPerPage);
  const paginatedResults = filteredResults.slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage);

  const handleExportCSV = () => {
    if (!data) return;
    const headers = ['meter_id', 'prediction', 'probability', 'risk_level', 'observation_count', 'missing_count', 'missing_ratio'];
    const rows = data.results.map((r) => [r.meter_id, r.prediction, r.probability, r.risk_level, r.observation_count, r.missing_count, r.missing_ratio]);
    
    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `gridbalance_batch_results_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-3">
          <Layers className="w-6 h-6 text-teal-400" />
          Batch Meter Tampering Analysis
        </h1>
        <p className="text-xs sm:text-sm text-slate-400 mt-1">
          Upload multi-meter CSV files for high-throughput automated tampering classification.
        </p>
      </div>

      {/* File Upload Box */}
      <div className="p-6 rounded-xl bg-slate-900 border border-slate-800 space-y-4">
        <h2 className="text-sm font-semibold text-white">Upload Multi-Meter Dataset</h2>
        <div className="flex flex-col sm:flex-row items-center gap-4">
          <label className="flex-1 flex items-center gap-3 p-3.5 border border-dashed border-slate-750 hover:border-teal-500/50 rounded-lg cursor-pointer bg-slate-950/50 transition-colors w-full">
            <Upload className="w-5 h-5 text-teal-400 shrink-0" />
            <span className="text-xs text-slate-300 truncate">
              {file ? file.name : 'Select CSV containing multiple customer meter profiles'}
            </span>
            <input type="file" accept=".csv" onChange={handleFileUpload} className="hidden" />
          </label>

          <button
            onClick={handleBatchProcess}
            disabled={loading || !file}
            className="w-full sm:w-auto px-6 py-3 rounded-lg bg-teal-600 hover:bg-teal-500 disabled:opacity-50 text-white font-medium text-xs sm:text-sm flex items-center justify-center gap-2 transition-all shadow-lg shadow-teal-950"
          >
            {loading ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                Batch Processing...
              </>
            ) : (
              <>
                <Layers className="w-4 h-4" />
                Run Batch Analysis
              </>
            )}
          </button>
        </div>

        {error && (
          <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* Results Table & Analytics */}
      {data && (
        <div className="space-y-6">
          {/* Summary Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
              <span className="text-xs text-slate-400 uppercase tracking-wider">Total Meters</span>
              <div className="text-2xl font-bold font-mono text-white mt-1">{data.total_meters}</div>
            </div>

            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
              <span className="text-xs text-slate-400 uppercase tracking-wider">Normal</span>
              <div className="text-2xl font-bold font-mono text-emerald-400 mt-1">{data.normal_count}</div>
            </div>

            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
              <span className="text-xs text-slate-400 uppercase tracking-wider">Potential Tampering</span>
              <div className="text-2xl font-bold font-mono text-rose-400 mt-1">{data.potential_tampering_count}</div>
            </div>

            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
              <span className="text-xs text-slate-400 uppercase tracking-wider">High Risk Cases</span>
              <div className="text-2xl font-bold font-mono text-amber-400 mt-1">{data.high_risk_count}</div>
            </div>
          </div>

          {/* Controls: Search, Filter, Export */}
          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs">
            <div className="flex items-center gap-3 w-full sm:w-auto">
              <div className="relative flex-1 sm:w-64">
                <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
                <input
                  type="text"
                  placeholder="Search Meter ID..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  className="w-full pl-9 pr-3 py-2 rounded-lg bg-slate-950 border border-slate-750 text-white focus:outline-none focus:border-teal-500"
                />
              </div>

              <div className="flex items-center gap-2">
                <Filter className="w-4 h-4 text-slate-400" />
                <select
                  value={filterRisk}
                  onChange={(e) => setFilterRisk(e.target.value)}
                  className="px-3 py-2 rounded-lg bg-slate-950 border border-slate-750 text-white focus:outline-none focus:border-teal-500"
                >
                  <option value="all">All Risks</option>
                  <option value="low">Low Risk</option>
                  <option value="moderate">Moderate Risk</option>
                  <option value="high">High Risk</option>
                  <option value="very high">Very High Risk</option>
                </select>
              </div>
            </div>

            <button
              onClick={handleExportCSV}
              className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 flex items-center gap-2 transition-colors shrink-0"
            >
              <Download className="w-4 h-4 text-teal-400" />
              Export Results CSV
            </button>
          </div>

          {/* Meter Results Table */}
          <div className="rounded-xl bg-slate-900 border border-slate-800 overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="bg-slate-950 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800">
                  <tr>
                    <th className="p-4">Meter ID</th>
                    <th className="p-4">Prediction</th>
                    <th className="p-4">Tampering Score</th>
                    <th className="p-4">Risk Level</th>
                    <th className="p-4">Missing Days</th>
                    <th className="p-4">Missing %</th>
                    <th className="p-4 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800">
                  {paginatedResults.map((item) => (
                    <tr
                      key={item.meter_id}
                      onClick={() => navigate(`/detection/${item.meter_id}`)}
                      className="hover:bg-slate-850 cursor-pointer transition-colors"
                    >
                      <td className="p-4 font-mono font-medium text-white">{item.meter_id}</td>
                      <td className="p-4">
                        {item.prediction === 'potential_tampering' ? (
                          <span className="text-rose-400 font-semibold flex items-center gap-1">
                            <AlertTriangle className="w-3.5 h-3.5" /> Tampering
                          </span>
                        ) : (
                          <span className="text-emerald-400 font-semibold flex items-center gap-1">
                            <CheckCircle2 className="w-3.5 h-3.5" /> Normal
                          </span>
                        )}
                      </td>
                      <td className="p-4 font-mono font-bold">{(item.probability * 100).toFixed(1)}%</td>
                      <td className="p-4">
                        <span className={`px-2 py-0.5 rounded-full text-[11px] font-semibold border ${
                          item.risk_level === 'Very High' ? 'text-rose-400 bg-rose-500/10 border-rose-500/30' :
                          item.risk_level === 'High' ? 'text-orange-400 bg-orange-500/10 border-orange-500/30' :
                          item.risk_level === 'Moderate' ? 'text-amber-400 bg-amber-500/10 border-amber-500/30' :
                          'text-emerald-400 bg-emerald-500/10 border-emerald-500/30'
                        }`}>
                          {item.risk_level}
                        </span>
                      </td>
                      <td className="p-4 font-mono">{item.missing_count}</td>
                      <td className="p-4 font-mono">{(item.missing_ratio * 100).toFixed(1)}%</td>
                      <td className="p-4 text-right">
                        <span className="text-teal-400 hover:underline text-[11px]">View Intelligence →</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination Controls */}
            <div className="p-4 bg-slate-950 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400">
              <span>Showing Page {currentPage} of {totalPages || 1}</span>
              <div className="flex items-center gap-2">
                <button
                  disabled={currentPage === 1}
                  onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
                  className="p-1.5 rounded bg-slate-900 border border-slate-750 disabled:opacity-50 text-slate-300"
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>
                <button
                  disabled={currentPage === totalPages || totalPages === 0}
                  onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
                  className="p-1.5 rounded bg-slate-900 border border-slate-750 disabled:opacity-50 text-slate-300"
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
