import React from 'react';
import { Grid } from '@mui/material';
import Card from '../components/Card';
import Map from '../components/Map';
import '../components/Content.css';

function MapPage() {
  return (
    <Grid container spacing={2}>
      <Grid item xs={12} sx={{ display: 'block' }}>
        <Card title="Map" className="card--bleed">
          <div className="map-slot">
            <Map />
          </div>
        </Card>
      </Grid>
    </Grid>
  );
}

export default MapPage;
