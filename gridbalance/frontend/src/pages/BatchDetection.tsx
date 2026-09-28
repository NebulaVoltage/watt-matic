import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Layers, Upload, Download, Search, AlertTriangle, ChevronLeft, ChevronRight, RefreshCw } from 'lucide-react';
import { api } from '../services/api';
import type { BatchMeterResponse } from '../services/api';

export const BatchDetection: React.FC = () => {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [batchData, setBatchData] = useState<BatchMeterResponse | null>(null);

  // Table filters & pagination
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [riskFilter, setRiskFilter] = useState<string>('all');
  const [currentPage, setCurrentPage] = useState<number>(1);
  const pageSize = 10;

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      setError("Please select a valid CSV file to upload.");
      return;
    }

    setError(null);
    setLoading(true);

    try {
      const res = await api.analyzeBatchCsv(file);
      setBatchData(res);
      setCurrentPage(1);

      // Save batch results map into session storage
      const historyStr = sessionStorage.getItem('gridbalance_meter_history') || '{}';
      const historyMap = JSON.parse(historyStr);
      res.results.forEach((r) => {
        historyMap[r.meter_id] = r;
      });
      sessionStorage.setItem('gridbalance_meter_history', JSON.stringify(historyMap));

      // Update session totals
      const prevAnalyzed = parseInt(sessionStorage.getItem('gridbalance_analyzed_count') || '0', 10);
      const prevFlagged = parseInt(sessionStorage.getItem('gridbalance_flagged_count') || '0', 10);
      sessionStorage.setItem('gridbalance_analyzed_count', (prevAnalyzed + res.total_meters).toString());
      sessionStorage.setItem('gridbalance_flagged_count', (prevFlagged + res.flagged_meters).toString());
    } catch (err: any) {
      setError(err.message || "Failed to process batch CSV upload.");
    } finally {
      setLoading(false);
    }
  };

  const handleExportCsv = () => {
    if (!batchData) return;
    const headers = ["meter_id", "prediction", "probability", "risk_level", "observation_count", "missing_count", "longest_missing_streak"];
    const rows = batchData.results.map(r => [
      r.meter_id,
      r.prediction,
      r.probability,
      r.risk_level,
      r.data_quality.observation_count,
      r.data_quality.missing_count,
      r.data_quality.longest_missing_streak
    ]);
    const csvContent = "data:text/csv;charset=utf-8," + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `GridBalance_Batch_Results_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const filteredResults = (batchData?.results || []).filter((r) => {
    const matchesSearch = r.meter_id.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesRisk =
      riskFilter === 'all' ||
      (riskFilter === 'flagged' && r.prediction === 'potential_tampering') ||
      (riskFilter === 'high' && r.risk_level === 'High') ||
      (riskFilter === 'normal' && r.prediction === 'normal');
    return matchesSearch && matchesRisk;
  });

  const totalPages = Math.ceil(filteredResults.length / pageSize) || 1;
  const paginatedResults = filteredResults.slice((currentPage - 1) * pageSize, currentPage * pageSize);

  const handleMeterClick = (meterId: string) => {
    navigate(`/detection/${encodeURIComponent(meterId)}`);
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl sm:text-3xl font-bold text-slate-100 font-mono tracking-tight flex items-center gap-3">
          <Layers className="w-7 h-7 text-emerald-400" />
          BATCH METER ANALYSIS
        </h1>
        <p className="text-slate-400 text-sm mt-1">
          Upload bulk electricity meter datasets to run parallel SGCC model classification and risk triage.
        </p>
      </div>

      {/* Upload Card */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
        <h2 className="text-sm font-mono uppercase tracking-wider text-slate-300 flex items-center gap-2">
          <Upload className="w-4 h-4 text-emerald-400" />
          Upload Bulk CSV Callset
        </h2>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="border-2 border-dashed border-slate-800 rounded-xl p-6 text-center hover:border-slate-700 transition-colors">
            <Upload className="w-8 h-8 text-slate-500 mx-auto mb-2" />
            <p className="text-xs font-mono text-slate-300">Drag & Drop CSV dataset file here</p>
            <p className="text-[11px] text-slate-500 font-mono mt-1">Requires 'meter_id' or 'CONS_NO' and daily consumption columns</p>
            <input
              type="file"
              accept=".csv, .txt"
              onChange={handleFileChange}
              className="mt-3 text-xs text-slate-400 file:mr-3 file:py-1.5 file:px-3 file:rounded-md file:border-0 file:text-xs file:font-mono file:bg-slate-800 file:text-emerald-400 cursor-pointer"
            />
          </div>

          {file && (
            <div className="text-xs font-mono text-emerald-400 bg-slate-950 p-2.5 rounded border border-slate-800 flex items-center justify-between">
              <span>Selected File: {file.name} ({(file.size / 1024).toFixed(1)} KB)</span>
              <button
                type="submit"
                disabled={loading}
                className="px-4 py-1.5 rounded bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold transition-colors cursor-pointer flex items-center gap-1"
              >
                {loading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : 'Run Batch Analysis'}
              </button>
            </div>
          )}

          {error && (
            <div className="p-3 bg-rose-950/40 border border-rose-800/60 rounded text-xs font-mono text-rose-300 flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
              <span>{error}</span>
            </div>
          )}
        </form>
      </div>

      {/* Batch Summary Cards — ONLY from actual API response */}
      {batchData && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
            <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl font-mono">
              <div className="text-xs text-slate-400 uppercase">Total Analyzed</div>
              <div className="text-2xl font-bold text-slate-100">{batchData.total_meters}</div>
            </div>

            <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl font-mono">
              <div className="text-xs text-slate-400 uppercase">Normal Meters</div>
              <div className="text-2xl font-bold text-emerald-400">{batchData.normal_meters}</div>
            </div>

            <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl font-mono">
              <div className="text-xs text-slate-400 uppercase">Potential Tampering</div>
              <div className="text-2xl font-bold text-rose-400">{batchData.flagged_meters}</div>
            </div>

            <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl font-mono">
              <div className="text-xs text-slate-400 uppercase">High Risk Cases</div>
              <div className="text-2xl font-bold text-amber-400">
                {batchData.results.filter(r => r.risk_level === 'High').length}
              </div>
            </div>
          </div>

          {/* Table Controls */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="relative">
                  <Search className="w-4 h-4 text-slate-500 absolute left-3 top-2.5" />
                  <input
                    type="text"
                    placeholder="Search Meter ID..."
                    value={searchTerm}
                    onChange={(e) => { setSearchTerm(e.target.value); setCurrentPage(1); }}
                    className="bg-slate-950 border border-slate-800 rounded-lg pl-9 pr-3 py-1.5 text-xs font-mono text-slate-200 focus:outline-none focus:border-slate-700"
                  />
                </div>

                <select
                  value={riskFilter}
                  onChange={(e) => { setRiskFilter(e.target.value); setCurrentPage(1); }}
                  className="bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs font-mono text-slate-300 focus:outline-none"
                >
                  <option value="all">All Predictions</option>
                  <option value="flagged">Potential Tampering Only</option>
                  <option value="high">High Risk Only</option>
                  <option value="normal">Normal Only</option>
                </select>
              </div>

              <button
                onClick={handleExportCsv}
                className="px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 text-xs font-mono transition-colors flex items-center gap-1.5 cursor-pointer"
              >
                <Download className="w-3.5 h-3.5" />
                Export CSV Results
              </button>
            </div>

            {/* Results Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-left font-mono text-xs border-collapse">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 uppercase text-[11px]">
                    <th className="py-2.5 px-3">Meter ID</th>
                    <th className="py-2.5 px-3">Prediction</th>
                    <th className="py-2.5 px-3">Probability</th>
                    <th className="py-2.5 px-3">Risk Level</th>
                    <th className="py-2.5 px-3">Data Quality</th>
                    <th className="py-2.5 px-3 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {paginatedResults.length > 0 ? (
                    paginatedResults.map((row) => (
                      <tr key={row.meter_id} className="hover:bg-slate-800/40 transition-colors">
                        <td className="py-2.5 px-3 font-bold text-slate-200">{row.meter_id}</td>
                        <td className="py-2.5 px-3">
                          <span className={`px-2 py-0.5 rounded text-[11px] font-semibold ${
                            row.prediction === 'potential_tampering'
                              ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                              : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                          }`}>
                            {row.prediction === 'potential_tampering' ? 'POTENTIAL TAMPERING' : 'NORMAL'}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 text-slate-300">{(row.probability * 100).toFixed(1)}%</td>
                        <td className="py-2.5 px-3">
                          <span className={`text-[11px] ${
                            row.risk_level === 'High' ? 'text-rose-400 font-bold' :
                            row.risk_level === 'Medium' ? 'text-amber-400 font-semibold' : 'text-slate-400'
                          }`}>
                            {row.risk_level}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 text-slate-400 text-[11px]">
                          {row.data_quality.observation_count} obs | {row.data_quality.missing_count} missing
                        </td>
                        <td className="py-2.5 px-3 text-right">
                          <button
                            onClick={() => handleMeterClick(row.meter_id)}
                            className="text-emerald-400 hover:underline font-semibold cursor-pointer"
                          >
                            Inspect Profile →
                          </button>
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={6} className="py-6 text-center text-slate-500 font-mono italic">
                        No meter results match the selected filter.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            {/* Pagination Controls */}
            <div className="flex items-center justify-between text-xs font-mono text-slate-400 pt-2 border-t border-slate-800">
              <span>Showing {filteredResults.length > 0 ? (currentPage - 1) * pageSize + 1 : 0} to {Math.min(currentPage * pageSize, filteredResults.length)} of {filteredResults.length} entries</span>
              <div className="flex items-center gap-2">
                <button
                  disabled={currentPage === 1}
                  onClick={() => setCurrentPage(p => Math.max(p - 1, 1))}
                  className="px-2.5 py-1 bg-slate-950 border border-slate-800 rounded disabled:opacity-40 hover:bg-slate-800 cursor-pointer"
                >
                  <ChevronLeft className="w-3.5 h-3.5" />
                </button>
                <span>Page {currentPage} of {totalPages}</span>
                <button
                  disabled={currentPage === totalPages}
                  onClick={() => setCurrentPage(p => Math.min(p + 1, totalPages))}
                  className="px-2.5 py-1 bg-slate-950 border border-slate-800 rounded disabled:opacity-40 hover:bg-slate-800 cursor-pointer"
                >
                  <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
