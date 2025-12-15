import React, { createContext, useState, useContext, useCallback, useRef } from 'react';

const ForecastContext = createContext();

export const useForecast = () => useContext(ForecastContext);

export const ForecastProvider = ({ children }) => {
  const [forecastData, setForecastData] = useState([]);
  const [historyData, setHistoryData] = useState([]);
  const [loadingForecast, setLoadingForecast] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [error, setError] = useState(null);
  
  // Use refs to track fetch status and time to avoid dependency cycles
  const lastFetchedForecast = useRef(0);
  const lastFetchedHistory = useRef(0);
  const isFetchingForecast = useRef(false);
  const isFetchingHistory = useRef(false);

  const fetchForecasts = useCallback(async (force = false) => {
    // Prevent concurrent fetches
    if (isFetchingForecast.current) return;

    // Cache check
    if (!force && (Date.now() - lastFetchedForecast.current < 3600000) && lastFetchedForecast.current > 0) {
      return;
    }

    isFetchingForecast.current = true;
    setLoadingForecast(true);
    setError(null);

    try {
      // 1. Fetch stations/overview (Init)
      await fetch('http://localhost:5000/api/init');
      
      // 2. Fetch all forecasts in one go
      // Demo date: 2018-07-15
      const res = await fetch('http://localhost:5000/api/predict_all?mode=forecast&date=2018-07-15');
      const data = await res.json();
      
      setForecastData(data.results || []);
      lastFetchedForecast.current = Date.now();
    } catch (err) {
      console.error("Error fetching forecast data:", err);
      setError(err.message);
    } finally {
      setLoadingForecast(false);
      isFetchingForecast.current = false;
    }
  }, []);

  const fetchHistory = useCallback(async (force = false) => {
    // Prevent concurrent fetches
    if (isFetchingHistory.current) return;

    // Cache check
    if (!force && (Date.now() - lastFetchedHistory.current < 3600000) && lastFetchedHistory.current > 0) {
      return;
    }

    isFetchingHistory.current = true;
    setLoadingHistory(true);
    setError(null);

    try {
      // 1. Fetch stations/overview (Init)
      await fetch('http://localhost:5000/api/init');

      // 2. Fetch all history values in one request
      // demo date: 2018-07-15
      const res = await fetch('http://localhost:5000/api/predict_all?mode=history&date=2018-07-15');
      const data = await res.json();
      
      setHistoryData(data.results || []);
      lastFetchedHistory.current = Date.now();
    } catch (err) {
      console.error("Error fetching history data:", err);
      setError(err.message);
    } finally {
      setLoadingHistory(false);
      isFetchingHistory.current = false;
    }
  }, []);

  const loading = loadingForecast || loadingHistory;

  return (
    <ForecastContext.Provider value={{ forecastData, historyData, loading, error, fetchForecasts, fetchHistory }}>
      {children}
    </ForecastContext.Provider>
  );
};
