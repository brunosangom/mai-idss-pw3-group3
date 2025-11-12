import React from 'react';
import Card from '../components/Card';
import '../components/Content.css';

function Forecast() {
  return (
    <div className="content-grid">
      <Card title="Forecast">
        <p>Weather Forecast & Predictions.</p>
      </Card>
    </div>
  );
}

export default Forecast;
