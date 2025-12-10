import React, { useEffect, useState } from 'react';
import './App.css';
import { Box, Container } from '@mui/material';
import Sidebar from './components/Sidebar';
import { Routes, Route } from 'react-router-dom';
import Overview from './pages/Overview';
import Statistics from './pages/Statistics';
import Forecast from './pages/Forecast';
import Historical from './pages/Historical';
import MapPage from './pages/MapPage';
import NotFound from './pages/NotFound';

function App() {
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    const mq = window.matchMedia('(max-width: 900px)');
    setCollapsed(mq.matches);
    const handler = (e) => setCollapsed(e.matches);
    mq.addEventListener ? mq.addEventListener('change', handler) : mq.addListener(handler);
    return () => {
      mq.removeEventListener ? mq.removeEventListener('change', handler) : mq.removeListener(handler);
    };
  }, []);

  return (
    <Box sx={{ display: 'flex', height: '100vh' }}>
      <Sidebar collapsed={collapsed} onToggle={() => setCollapsed(c => !c)} />
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 2,
          ml: collapsed ? '56px' : '200px',
          transition: 'margin-left 0.2s ease',
          overflow: 'auto',
          backgroundColor: '#ffffff',
        }}
      >
        <Container maxWidth={false} disableGutters>
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/statistics" element={<Statistics />} />
            <Route path="/forecast" element={<Forecast />} />
            <Route path="/historical" element={<Historical />} />
            <Route path="/map" element={<MapPage />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </Container>
      </Box>
    </Box>
  );
}

export default App;