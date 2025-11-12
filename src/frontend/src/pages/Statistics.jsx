import React from 'react';
import Card from '../components/Card';
import FireTrendChart from '../components/charts/FireTrendChart';
import FireComparisonChart from '../components/charts/FireComparisonChart';
import SeasonalityChart from '../components/charts/SeasonalityChart';
import '../components/Content.css';

function Statistics() {
  return (
    <div className="content-grid">
      <Card title="Weekly Wildfire Trend">
        <FireTrendChart />
      </Card>

      <Card title="Yearly Comparison">
        <FireComparisonChart />
      </Card>

      <Card title="Monthly Seasonality">
        <SeasonalityChart />
      </Card>

      <Card title="Notes">
        <div className="stack">
          <p>Wildfire activity increases from spring to late summer; containment usually lags by 2–4 weeks.</p>
          <ul className="list list--compact">
            <li className="list-item">Hotspots: West and South regions.</li>
            <li className="list-item">Resources: Aerial support at 75% utilization.</li>
            <li className="list-item">Forecast: Elevated risk next 10 days.</li>
          </ul>
        </div>
      </Card>
    </div>
  );
}

export default Statistics;
