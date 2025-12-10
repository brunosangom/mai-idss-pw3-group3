import React from 'react';
import { Grid } from '@mui/material';
import Card from '../components/Card';
import '../components/Content.css';

function Historical() {
  return (
    <Grid container spacing={2}>
      <Grid item xs={12} sx={{ display: 'block' }}>
        <Card title="Historical Data">
          <p>Historical data analysis.</p>
        </Card>
      </Grid>
    </Grid>
  );
}

export default Historical;
