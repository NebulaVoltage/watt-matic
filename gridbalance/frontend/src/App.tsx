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
      <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100 selection:bg-teal-500 selection:text-slate-950">
        <Navbar />
        <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
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
