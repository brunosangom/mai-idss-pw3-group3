import React from 'react';
import { Grid } from '@mui/material';
import Card from '../components/Card';
import '../components/Content.css';
import Map from '../components/Map';

function Forecast() {
  return (
    <Grid container spacing={2}>
      <Grid item xs={12}>
        <Card title="Forecast - Fire Weather Index">
          <div className="map-slot">
            <Map wide={true} />
          </div>
        </Card>
      </Grid>
    </Grid>
  );
}

export default Forecast;
