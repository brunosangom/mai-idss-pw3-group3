import React, { useMemo, useState } from 'react';
import { Button, Typography, Collapse } from '@mui/material';
import Card from '../components/Card';
import ForecastOverviewChart from '../components/charts/ForecastOverviewChart';
import ForecastDistributionChart from '../components/charts/ForecastDistributionChart';
import { useForecast } from '../contexts/ForecastContext';
import '../components/Content.css';

function Statistics() {
  const { forecastData, historyData, fetchForecasts, fetchHistory, loadingForecast, loadingHistory } = useForecast();
  const [notesExpanded, setNotesExpanded] = useState(true);

  const processData = (data, type) => {
    if (!data || data.length === 0) return { overviewData: [], distributionData: [] };

    const isForecast = type === 'forecast';
    const key = isForecast ? 'forecast' : 'history';
    const riskLevels = ['Low', 'Moderate', 'High', 'Extreme'];
    
    const daysMap = {};
    const distribution = {};
    riskLevels.forEach(level => distribution[level] = 0);

    data.forEach(station => {
      const list = station[key];
      if (list) {
        list.forEach((item, index) => {
            let dayLabel;
            if (isForecast) {
                // Forecast: 0 -> Today, 1 -> +1
                dayLabel = index === 0 ? 'Today' : `+${index}`;
            } else {
                // History: 0 -> -7, ..., 7 -> Today
                // Based on app.py: range(7, -1, -1) -> 7, 6, ..., 0
                // So index 0 corresponds to offset 7 (7 days ago)
                // index 7 corresponds to offset 0 (Today)
                const offset = 7 - index;
                dayLabel = offset === 0 ? 'Today' : `-${offset}`;
            }
            
            if (!daysMap[dayLabel]) {
                daysMap[dayLabel] = { day: dayLabel };
                riskLevels.forEach(l => daysMap[dayLabel][l] = 0);
            }
            
            const risk = item.risk_level;
            if (riskLevels.includes(risk)) {
                daysMap[dayLabel][risk]++;
                distribution[risk]++;
            }
        });
      }
    });
    
    let overviewData = Object.values(daysMap);
    
    // Sort
    overviewData.sort((a, b) => {
        const getVal = (d) => {
            if (d === 'Today') return 0;
            return parseInt(d);
        };
        return getVal(a.day) - getVal(b.day);
    });

    const distributionData = Object.keys(distribution).map(k => ({
        id: k,
        label: k,
        value: distribution[k]
    })).filter(d => d.value > 0);
    
    return { overviewData, distributionData };
  };

  const forecastStats = useMemo(() => processData(forecastData, 'forecast'), [forecastData]);
  const historyStats = useMemo(() => processData(historyData, 'history'), [historyData]);

  const renderSection = (title, data, stats, onFetch, fetchLabel, isLoading) => {
    const hasData = data && data.length > 0;

    if (!hasData) {
      return (
        <Card title={title}>
          <div style={{ display: 'flex', justifyContent: 'center', padding: '20px' }}>
            <Button 
                variant="contained" 
                onClick={() => onFetch()} 
                disabled={isLoading}
            >
                {isLoading ? 'Loading...' : fetchLabel}
            </Button>
          </div>
        </Card>
      );
    }

    return (
      <>
        <Card title={`${title} Overview`}>
            <ForecastOverviewChart data={stats.overviewData} />
        </Card>
        <Card title={`${title} Risk Distribution`}>
            <ForecastDistributionChart data={stats.distributionData} />
        </Card>
      </>
    );
  };

  return (
    <div className="content-grid">
      <Card 
          title="Dashboard Guide & Methodology" 
          actions={
              <Button size="small" onClick={() => setNotesExpanded(!notesExpanded)}>
                  {notesExpanded ? 'Hide' : 'Show'}
              </Button>
          }
      >
        <Collapse in={notesExpanded}>
          <div className="stack">
              <Typography variant="body2" paragraph>
                  This dashboard provides a comprehensive view of wildfire risks across monitored stations, split into <strong>Forecast</strong> (next 7 days) and <strong>History</strong> (past 7 days).
              </Typography>
              <Typography variant="subtitle2" gutterBottom>Risk Calculation Methodology:</Typography>
              <Typography variant="body2" paragraph>
                  The risk levels (<strong>Low, Moderate, High, Extreme</strong>) are computed using a hybrid model:
              </Typography>
              <ul className="list list--compact">
                  <li className="list-item"><strong>Machine Learning:</strong> A predictive model analyzes historical patterns and current weather conditions.</li>
                  <li className="list-item"><strong>Fire Weather Index (FWI):</strong> Standard meteorological indices are calculated to validate and refine the ML predictions.</li>
              </ul>
              <Typography variant="body2" sx={{ mt: 1 }}>
                  The <strong>Overview</strong> charts show the count of stations at each risk level per day, while the <strong>Distribution</strong> charts show the aggregate share of risk levels for the selected period.
              </Typography>
          </div>
        </Collapse>
      </Card>

      {renderSection("Forecast", forecastData, forecastStats, fetchForecasts, "Fetch Forecast Data", loadingForecast)}
      {renderSection("History", historyData, historyStats, fetchHistory, "Fetch Historical Data", loadingHistory)}
    </div>
  );
}

export default Statistics;
