import React from 'react';
import { ShieldCheck, Info } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="border-t border-slate-800 bg-slate-950 py-8 text-xs text-slate-400">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row justify-between items-center gap-4">
        <div>
          <p className="font-semibold text-slate-200">GridBalance Smart Grid Intelligence Platform</p>
          <p className="text-slate-500 mt-1">Optimizing Smart Grids through Regression-Based Load Forecasting and Meter Tampering Classification.</p>
        </div>

        <div className="flex items-center gap-6 text-slate-400">
          <div className="flex items-center gap-1.5">
            <ShieldCheck className="w-4 h-4 text-teal-400" />
            <span>SGCC & UCI Frozen ML Engines</span>
          </div>
          <div className="flex items-center gap-1.5 text-slate-500">
            <Info className="w-4 h-4" />
            <span>Research Evaluation Benchmark</span>
          </div>
        </div>
      </div>
    </footer>
  );
};
