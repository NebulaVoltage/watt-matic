import React from 'react';
import { ShieldCheck, ExternalLink } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="bg-slate-950 border-t border-slate-800 text-slate-400 py-8 px-4 sm:px-6 lg:px-8 mt-auto">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4 text-xs">
        <div className="flex items-center gap-2 font-mono">
          <ShieldCheck className="w-4 h-4 text-emerald-400" />
          <span className="text-slate-200">GridBalance Research Platform</span>
          <span className="text-slate-600">|</span>
          <span>SGCC 18-Feature Classifier & UCI 29-Feature Regressor</span>
        </div>

        <div className="flex items-center gap-4 font-mono">
          <a
            href="http://127.0.0.1:8000/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-emerald-400 transition-colors flex items-center gap-1"
          >
            FastAPI Swagger Docs <ExternalLink className="w-3 h-3" />
          </a>
          <span className="text-slate-600">|</span>
          <span>Sealed Test Evaluation Verified</span>
        </div>
      </div>
    </footer>
  );
};
