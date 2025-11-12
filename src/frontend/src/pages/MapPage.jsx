import React from 'react';
import Card from '../components/Card';
import Map from '../components/Map';
import '../components/Content.css';

function MapPage() {
  return (
    <div className="content-grid">
      <Card title="Map" className="card--bleed">
        <div className="map-slot">
          <Map />
        </div>
      </Card>
    </div>
  );
}

export default MapPage;
