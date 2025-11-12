import React from 'react';
import './App.css';
import Sidebar from './components/Sidebar';
import { Routes, Route } from 'react-router-dom';
import Overview from './pages/Overview';
import Statistics from './pages/Statistics';
import Forecast from './pages/Forecast';
import Historical from './pages/Historical';
import MapPage from './pages/MapPage';
import NotFound from './pages/NotFound';

function App() {
  return (
    <div className="app-layout">
      <Sidebar />
      <main className="app-main">
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/statistics" element={<Statistics />} />
          <Route path="/forecast" element={<Forecast />} />
          <Route path="/historical" element={<Historical />} />
          <Route path="/map" element={<MapPage />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;