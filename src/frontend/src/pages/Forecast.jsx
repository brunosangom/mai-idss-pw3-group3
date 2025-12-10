import React from 'react';
import { Grid } from '@mui/material';
import Card from '../components/Card';
import '../components/Content.css';

function Forecast() {
  return (
    <Grid container spacing={2}>
      <Grid item xs={12} sx={{ display: 'block' }}>
        <Card title="Forecast - Fire Weather Index" />
      </Grid>
    </Grid>
  );
}

export default Forecast;
