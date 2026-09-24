import React, { useEffect, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Activity, ShieldAlert, LineChart, Cpu, BookOpen, Layers, AlertTriangle } from 'lucide-react';
import { fetchHealth, type HealthStatus } from '../services/api';

export const Navbar: React.FC = () => {
  const location = useLocation();
  const [health, setHealth] = useState<HealthStatus | null>(null);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  const navLinks = [
    { path: '/', label: 'Overview', icon: Activity },
    { path: '/detection', label: 'Tampering Detection', icon: ShieldAlert },
    { path: '/detection/batch', label: 'Batch Analysis', icon: Layers },
    { path: '/forecast', label: 'Load Forecast', icon: LineChart },
    { path: '/models', label: 'Model Intelligence', icon: Cpu },
    { path: '/research', label: 'Research & Methodology', icon: BookOpen },
  ];

  return (
    <header className="sticky top-0 z-50 bg-slate-900/90 backdrop-blur-md border-b border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <Link to="/" className="flex items-center gap-3">
            <div className="p-2 bg-teal-500/10 border border-teal-500/30 rounded-lg">
              <Activity className="w-5 h-5 text-teal-400" />
            </div>
            <div>
              <span className="font-bold text-lg text-white tracking-wider">GRIDBALANCE</span>
              <span className="block text-[10px] text-teal-400 font-mono font-medium uppercase tracking-widest">Smart Grid Platform</span>
            </div>
          </Link>

          <nav className="hidden md:flex items-center space-x-1">
            {navLinks.map((link) => {
              const Icon = link.icon;
              const isActive = location.pathname === link.path || (link.path !== '/' && location.pathname.startsWith(link.path) && link.path !== '/detection');
              return (
                <Link
                  key={link.path}
                  to={link.path}
                  className={`px-3 py-2 rounded-md text-xs font-medium transition-colors flex items-center gap-2 ${
                    isActive
                      ? 'bg-teal-500/10 text-teal-400 border border-teal-500/30'
                      : 'text-slate-300 hover:bg-slate-800 hover:text-white'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  {link.label}
                </Link>
              );
            })}
          </nav>

          <div className="flex items-center gap-3">
            <div className="hidden sm:flex items-center gap-2 text-xs px-3 py-1.5 rounded-full bg-slate-850 border border-slate-750">
              <span className="text-slate-400 font-mono">Engines:</span>
              <div className="flex items-center gap-1.5">
                {health?.sgcc_model === 'loaded' ? (
                  <span className="flex items-center gap-1 text-emerald-400 font-medium">
                    <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span> SGCC
                  </span>
                ) : (
                  <span className="text-rose-400 flex items-center gap-1">
                    <AlertTriangle className="w-3 h-3" /> SGCC
                  </span>
                )}
                <span className="text-slate-600">|</span>
                {health?.forecast_model === 'loaded' ? (
                  <span className="flex items-center gap-1 text-emerald-400 font-medium">
                    <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span> UCI
                  </span>
                ) : (
                  <span className="text-rose-400 flex items-center gap-1">
                    <AlertTriangle className="w-3 h-3" /> UCI
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};
