import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Navbar } from './components/Navbar';
import { Footer } from './components/Footer';
import { OverviewDashboard } from './pages/OverviewDashboard';
import { MeterDetection } from './pages/MeterDetection';
import { BatchDetection } from './pages/BatchDetection';
import { MeterIntelligence } from './pages/MeterIntelligence';
import { LoadForecasting } from './pages/LoadForecasting';
import { ModelIntelligence } from './pages/ModelIntelligence';
import { ResearchMethodology } from './pages/ResearchMethodology';

export const App: React.FC = () => {
  return (
    <Router>
      <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-emerald-500/30 selection:text-emerald-300">
        <div className="m-4 p-4 bg-emerald-600 border-2 border-emerald-300 rounded-xl text-emerald-100 text-2xl font-bold text-center shadow-lg" id="css-test-banner">
          [CSS PIPELINE VERIFIED ACTIVE] Tailwind v4 Utility Test Rendered Successfully
        </div>
        <Navbar />
        <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <Routes>
            <Route path="/" element={<OverviewDashboard />} />
            <Route path="/detection" element={<MeterDetection />} />
            <Route path="/detection/batch" element={<BatchDetection />} />
            <Route path="/detection/:meterId" element={<MeterIntelligence />} />
            <Route path="/forecast" element={<LoadForecasting />} />
            <Route path="/models" element={<ModelIntelligence />} />
            <Route path="/research" element={<ResearchMethodology />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
};

export default App;
