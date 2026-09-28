import React, { useState, useEffect } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { Activity, ShieldAlert, Cpu, FileText, Search, Zap } from 'lucide-react';
import { api } from '../services/api';
import type { HealthResponse } from '../services/api';

export const Navbar: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loadingHealth, setLoadingHealth] = useState<boolean>(true);

  useEffect(() => {
    let isMounted = true;
    const checkStatus = async () => {
      const statusRes = await api.checkHealth();
      if (isMounted) {
        setHealth(statusRes);
        setLoadingHealth(false);
      }
    };
    checkStatus();
    const interval = setInterval(checkStatus, 10000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const navItems = [
    { name: 'Overview', path: '/', icon: Activity },
    { name: 'Detection', path: '/detection', icon: ShieldAlert },
    { name: 'Forecasting', path: '/forecast', icon: Zap },
    { name: 'Model Intelligence', path: '/models', icon: Cpu },
    { name: 'Research', path: '/research', icon: FileText },
  ];

  const isOnline = health?.status === 'healthy';

  return (
    <header className="sticky top-0 z-50 bg-slate-950/90 backdrop-blur-md border-b border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        
        {/* Brand */}
        <div className="flex items-center gap-3">
          <Link to="/" className="flex items-center gap-2.5 group">
            <div className="w-9 h-9 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400 group-hover:border-emerald-400 transition-colors">
              <Zap className="w-5 h-5" />
            </div>
            <div>
              <span className="text-lg font-bold tracking-wider text-slate-100 font-mono">GRIDBALANCE</span>
              <span className="hidden sm:block text-[10px] text-slate-400 uppercase tracking-widest font-mono">Smart Grid Intelligence</span>
            </div>
          </Link>
        </div>

        {/* Navigation Links */}
        <nav className="hidden md:flex items-center gap-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path || (item.path !== '/' && location.pathname.startsWith(item.path));
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors flex items-center gap-1.5 ${
                  isActive
                    ? 'bg-slate-800 text-emerald-400 border border-slate-700'
                    : 'text-slate-300 hover:text-slate-100 hover:bg-slate-900'
                }`}
              >
                <Icon className="w-4 h-4 opacity-80" />
                {item.name}
              </Link>
            );
          })}
        </nav>

        {/* Status Badge & Primary Action */}
        <div className="flex items-center gap-3">
          {/* Real Backend Status Indicator */}
          <div className={`px-2.5 py-1 rounded-full border text-xs font-mono flex items-center gap-1.5 ${
            isOnline
              ? 'bg-emerald-950/40 border-emerald-800/60 text-emerald-400'
              : 'bg-rose-950/40 border-rose-800/60 text-rose-400'
          }`}>
            <span className={`w-2 h-2 rounded-full ${isOnline ? 'bg-emerald-400 animate-pulse' : 'bg-rose-500'}`}></span>
            <span>{loadingHealth ? 'Checking Engine...' : isOnline ? 'ML Engine Online' : 'ML Engine Offline'}</span>
          </div>

          {/* Primary Action Button */}
          <button
            onClick={() => navigate('/detection')}
            className="px-3.5 py-1.5 rounded-md bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-semibold text-xs tracking-wide uppercase font-mono shadow-sm transition-colors flex items-center gap-1.5 cursor-pointer"
          >
            <Search className="w-3.5 h-3.5" />
            Analyze Meter
          </button>
        </div>
      </div>
    </header>
  );
};
