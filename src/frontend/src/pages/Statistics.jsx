import React from 'react';
import { Grid } from '@mui/material';
import Card from '../components/Card';
import FireTrendChart from '../components/charts/FireTrendChart';
import FireComparisonChart from '../components/charts/FireComparisonChart';
import SeasonalityChart from '../components/charts/SeasonalityChart';
import '../components/Content.css';

function Statistics() {
  return (
    <Grid container spacing={2} sx={{ width: '100%' }}>
      <Grid item xs={12} md={4} sx={{ flex: { xs: '1 0 100%', md: 1 }, maxWidth: 'none' }}>
        <Card title="Weekly Wildfire Trend">
          <FireTrendChart />
        </Card>
      </Grid>

      <Grid item xs={12} md={4} sx={{ flex: { xs: '1 0 100%', md: 1 }, maxWidth: 'none' }}>
        <Card title="Yearly Comparison">
          <FireComparisonChart />
        </Card>
      </Grid>

      <Grid item xs={12} md={4} sx={{ flex: { xs: '1 0 100%', md: 1 }, maxWidth: 'none' }}>
        <Card title="Monthly Seasonality">
          <SeasonalityChart />
        </Card>
      </Grid>

      <Grid item xs={12} sx={{ flex: { xs: '1 0 100%', md: '1 0 100%' }, maxWidth: 'none' }}>
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
      </Grid>
    </Grid>
  );
}

export default Statistics;
