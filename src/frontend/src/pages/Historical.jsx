import React, { useEffect, useState } from 'react';
import { Grid, Typography, CircularProgress, Alert, Button, Collapse, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper } from '@mui/material';
import Card from '../components/Card';
import WeatherMetricChart from '../components/charts/WeatherMetricChart';
import PredictionAccuracyChart from '../components/charts/PredictionAccuracyChart';
import ErrorRatesChart from '../components/charts/ErrorRatesChart';
import { useForecast } from '../contexts/ForecastContext';
import '../components/Content.css';

function Historical() {
  const { historicalAnalysisData, fetchHistoricalAnalysis, loadingHistoricalAnalysis, error } = useForecast();
  const [notesExpanded, setNotesExpanded] = useState(true);

  useEffect(() => {
    fetchHistoricalAnalysis();
  }, [fetchHistoricalAnalysis]);

  if (loadingHistoricalAnalysis && !historicalAnalysisData) {
    return (
        <Grid container justifyContent="center" sx={{ mt: 4 }}>
            <CircularProgress />
        </Grid>
    );
  }

  if (error && !historicalAnalysisData) {
    return (
        <Grid container justifyContent="center" sx={{ mt: 4 }}>
            <Alert severity="error">{error}</Alert>
        </Grid>
    );
  }

  const data = historicalAnalysisData || { weather_trends: [], prediction_accuracy: [], error_trends: [], hardest_stations: [] };

  return (
    <Grid container spacing={2} sx={{ width: '100%' }}>

      <Grid item xs={12} sx={{ flex: { xs: '1 0 100%', md: '1 0 100%' }, maxWidth: 'none' }}>
        <Card 
            title="Historical Analysis Guide" 
            actions={
                <Button size="small" onClick={() => setNotesExpanded(!notesExpanded)}>
                    {notesExpanded ? 'Hide' : 'Show'}
                </Button>
            }
        >
          <Collapse in={notesExpanded}>
            <div className="stack">
                <Typography variant="body1" paragraph>
                    Analysis of weather trends and model performance based on historical data.
                </Typography>
                <Typography variant="body2" paragraph>
                    <strong>Weather Trends:</strong> Displays the Min, Max, and Average values for Temperature, Precipitation, and Wind Speed over the years.
                </Typography>
                <Typography variant="body2" paragraph>
                    <strong>Prediction Accuracy:</strong> Tracks the performance of the wildfire prediction model across selected stations.
                </Typography>
                <Typography variant="body2" paragraph>
                    <strong>Error Rates:</strong> Shows the trends of False Positives (predicting fire when there is none) and False Negatives (failing to predict a fire).
                </Typography>
            </div>
          </Collapse>
        </Card>
      </Grid>

      <Grid item xs={12} md={4} sx={{ flex: { xs: '1 0 100%', md: 1 }, maxWidth: 'none' }}>
        <Card title="Weather Trends: Temperature">
            <Typography variant="body2" color="textSecondary" gutterBottom>
                Min, Max, and Average Max Temperature (K).
            </Typography>
            <WeatherMetricChart 
                data={data.weather_trends} 
                metric="tmmx" 
                title="" 
                yLabel="Temp (K)"
            />
        </Card>
      </Grid>

      <Grid item xs={12} md={4} sx={{ flex: { xs: '1 0 100%', md: 1 }, maxWidth: 'none' }}>
        <Card title="Weather Trends: Precipitation">
            <Typography variant="body2" color="textSecondary" gutterBottom>
                Min, Max, and Average Precipitation (mm).
            </Typography>
            <WeatherMetricChart 
                data={data.weather_trends} 
                metric="pr" 
                title="" 
                yLabel="Precipitation (mm)"
            />
        </Card>
      </Grid>

      <Grid item xs={12} md={4} sx={{ flex: { xs: '1 0 100%', md: 1 }, maxWidth: 'none' }}>
        <Card title="Weather Trends: Wind Speed">
            <Typography variant="body2" color="textSecondary" gutterBottom>
                Min, Max, and Average Wind Speed (m/s).
            </Typography>
            <WeatherMetricChart 
                data={data.weather_trends} 
                metric="vs" 
                title="" 
                yLabel="Wind Speed (m/s)"
            />
        </Card>
      </Grid>

      <Grid item xs={12} sx={{ flex: { xs: '1 0 100%', md: '1 0 100%' }, maxWidth: 'none' }}>
        <Card title="Prediction Accuracy Trends">
            <Typography variant="body2" color="textSecondary" gutterBottom>
                Model accuracy over time for selected stations.
            </Typography>
            <PredictionAccuracyChart data={data.prediction_accuracy} />
        </Card>
      </Grid>

      <Grid item xs={12} sx={{ flex: { xs: '1 0 100%', md: '1 0 100%' }, maxWidth: 'none' }}>
        <Card title="Error Rate Trends">
            <Typography variant="body2" color="textSecondary" gutterBottom>
                False Positive and False Negative rates over time.
            </Typography>
            <ErrorRatesChart data={data.error_trends} />
        </Card>
      </Grid>

      <Grid item xs={12} sx={{ flex: { xs: '1 0 100%', md: '1 0 100%' }, maxWidth: 'none' }}>
        <Card title="Hardest to Predict Stations">
            <Typography variant="body2" color="textSecondary" gutterBottom>
                Stations with the lowest overall prediction accuracy.
            </Typography>
            <TableContainer component={Paper} sx={{ boxShadow: 'none', border: '1px solid #e0e0e0' }}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Station Name</TableCell>
                    <TableCell align="right">Accuracy (%)</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {data.hardest_stations && data.hardest_stations.map((row) => (
                    <TableRow key={row.name}>
                      <TableCell component="th" scope="row">
                        {row.name}
                      </TableCell>
                      <TableCell align="right">{row.accuracy}%</TableCell>
                    </TableRow>
                  ))}
                  {(!data.hardest_stations || data.hardest_stations.length === 0) && (
                      <TableRow>
                          <TableCell colSpan={2} align="center">No data available</TableCell>
                      </TableRow>
                  )}
                </TableBody>
              </Table>
            </TableContainer>
        </Card>
      </Grid>
    </Grid>
  );
}

export default Historical;
