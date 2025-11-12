import React, { useState } from 'react';
import './Content.css';
import Card from './Card';
import Map from './Map';

function Content() {
  const [wideMap, setWideMap] = useState(false);
  const toggleWide = () => setWideMap(w => !w);

  return (
    <div className="content-grid overview-grid">
      <Card
        title="Map"
        className={`map-card ${wideMap ? 'map-card--wide card--bleed' : ''}`}
        actions={(
          <button
            type="button"
            className="map-expand-btn"
            aria-label={wideMap ? 'Collapse map width' : 'Expand map width'}
            onClick={toggleWide}
          >
            {wideMap ? 'Collapse' : 'Expand'}
          </button>
        )}
      >
        <div className="map-slot">
          <Map wide={wideMap} />
        </div>
      </Card>

      <Card title="Fire Situation">
        <div className="mini-kpi-grid">
          <div className="mini-kpi mini-kpi--critical">
            <div className="mini-kpi-label">Active Fires</div>
            <div className="mini-kpi-value">42</div>
          </div>
          <div className="mini-kpi mini-kpi--warning">
            <div className="mini-kpi-label">Acres Burned (Today)</div>
            <div className="mini-kpi-value">8,120</div>
          </div>
          <div className="mini-kpi mini-kpi--neutral">
            <div className="mini-kpi-label">Avg Containment</div>
            <div className="mini-kpi-value">57%</div>
          </div>
          <div className="mini-kpi mini-kpi--neutral">
            <div className="mini-kpi-label">Median Response</div>
            <div className="mini-kpi-value">21m</div>
          </div>
          <div className="mini-kpi mini-kpi--alert">
            <div className="mini-kpi-label">Evac Alerts</div>
            <div className="mini-kpi-value">5</div>
          </div>
        </div>
      </Card>

      <Card title="Environment & Resources">
        <div className="mini-kpi-grid">
          <div className="mini-kpi mini-kpi--neutral">
            <div className="mini-kpi-label">Resource Utilization</div>
            <div className="mini-kpi-value">76%</div>
          </div>
          <div className="mini-kpi mini-kpi--risk">
            <div className="mini-kpi-label">Weather Risk</div>
            <div className="mini-kpi-value">High</div>
          </div>
          <div className="mini-kpi mini-kpi--warning">
            <div className="mini-kpi-label">Air Quality (PM2.5)</div>
            <div className="mini-kpi-value">132</div>
          </div>
          <div className="mini-kpi mini-kpi--neutral">
            <div className="mini-kpi-label">Crew Deployments</div>
            <div className="mini-kpi-value">18</div>
          </div>
          <div className="mini-kpi mini-kpi--neutral">
            <div className="mini-kpi-label">Aerial Missions</div>
            <div className="mini-kpi-value">11</div>
          </div>
        </div>
      </Card>
    </div>
  );
}

export default Content;
