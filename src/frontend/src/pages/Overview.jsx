import React, { useState, useEffect, useMemo } from 'react';
import { Grid, List, ListItem, ListItemButton, ListItemText, ListItemIcon, Divider, Typography, Button, Dialog, DialogTitle, DialogContent, DialogActions, Slider, Box, CircularProgress } from '@mui/material';
import { MdLocalFireDepartment, MdWbSunny, MdWaterDrop, MdInfo, MdWarning } from 'react-icons/md';
import '../components/Content.css';
import Card from '../components/Card';
import InteractiveMap from '../components/InteractiveMap';
import { useForecast } from '../contexts/ForecastContext';

function Overview() {
  const [wideMap, setWideMap] = useState(false);
  const { historyData, forecastData, loading, fetchHistory, fetchForecasts } = useForecast();
  
  const [openWarningDialog, setOpenWarningDialog] = useState(false);
  const [warningRadius, setWarningRadius] = useState(10);

  useEffect(() => {
    fetchHistory();
    fetchForecasts();
  }, [fetchHistory, fetchForecasts]);

  const predictionSummary = useMemo(() => {
    if (!forecastData || forecastData.length === 0) return { high: 0, medium: 0, low: 0, total: 0 };
    
    let high = 0;
    let medium = 0;
    let low = 0;

    forecastData.forEach(station => {
        if (station.forecast && station.forecast.length > 0) {
            const risks = station.forecast.map(f => f.risk_level);
            if (risks.includes('High') || risks.includes('Extreme') || risks.includes('Very High')) high++;
            else if (risks.includes('Moderate')) medium++;
            else low++;
        }
    });

    return { high, medium, low, total: forecastData.length };
  }, [forecastData]);

  const handleIssueWarning = () => {
    console.log(`Issuing warning with radius ${warningRadius}km`);
    setOpenWarningDialog(false);
  };

  // Transform historyData for InteractiveMap
  // historyData: [{ name, location, history: [{ date, risk_level, ... }] }]
  // InteractiveMap expects: [{ name, lat, lon, data: { [offset]: { risk_level, ... } } }]
  
  const mapData = historyData.map(station => {
    const dataMap = {};
    if (station.history) {
      station.history.forEach((day, index) => {
        // Calculate offset from today
        // The backend returns T-7 to T-0.
        const date = new Date(day.date);
        const today = new Date('2018-07-15');
        // Reset time part
        date.setHours(0,0,0,0);
        today.setHours(0,0,0,0);
        
        const diffTime = date - today;
        const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
        
        dataMap[diffDays] = day;
      });
    }
    return {
      name: station.name,
      lat: station.location.latitude,
      lon: station.location.longitude,
      data: dataMap
    };
  });

  return (
    <Grid container spacing={2} sx={{ width: '100%' }}>
      <Grid item xs={12} md={wideMap ? 8 : 6} sx={{ flex: { xs: '1 0 100%', md: 2 }, maxWidth: 'none' }}>
        <InteractiveMap 
          title="Historical Fire Risk (Last 7 Days)"
          stationsData={mapData}
          timeRange={{ start: -7, end: 0 }}
          wide={wideMap}
          onToggleWide={() => setWideMap(w => !w)}
          loading={loading}
          referenceDate={new Date('2018-07-15')}
        />
      </Grid>

      <Grid item xs={12} md={wideMap ? 4 : 3} sx={{ flex: { xs: '1 0 100%', md: 1 }, maxWidth: 'none' }}>
        <Card title="Prediction Summary">
          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
              <CircularProgress />
            </Box>
          ) : (
            <>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <div className="mini-kpi mini-kpi--critical">
                    <div className="mini-kpi-label">High Risk Stations</div>
                    <div className="mini-kpi-value">{predictionSummary.high}</div>
                </div>
                <div className="mini-kpi mini-kpi--warning">
                    <div className="mini-kpi-label">Medium Risk Stations</div>
                    <div className="mini-kpi-value">{predictionSummary.medium}</div>
                </div>
                <div className="mini-kpi mini-kpi--neutral">
                    <div className="mini-kpi-label">Low Risk Stations</div>
                    <div className="mini-kpi-value">{predictionSummary.low}</div>
                </div>
              </div>
              
              <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center' }}>
                <Button 
                    variant="contained" 
                    color="error" 
                    startIcon={<MdWarning />}
                    onClick={() => setOpenWarningDialog(true)}
                >
                    Issue Warning
                </Button>
              </Box>
            </>
          )}
        </Card>
      </Grid>

      <Grid item xs={12} md={wideMap ? 12 : 3} sx={{ flex: { xs: '1 0 100%', md: wideMap ? '1 0 100%' : 1 }, maxWidth: 'none' }}>
      <Card title="Other Resources">
        <List dense>
          <ListItem disablePadding>
            <ListItemButton component="a" href="https://www.ready.gov/wildfires" target="_blank">
              <ListItemIcon>
                <MdInfo size={24} color="#2f3ad3ff" />
              </ListItemIcon>
              <ListItemText 
                primary="ready.gov Wildfires" 
                secondary="Preparedness & Safety"
              />
            </ListItemButton>
          </ListItem>
        </List>
        <Divider sx={{ my: 1 }} />
        <Typography variant="subtitle2" sx={{ px: 2, pt: 1, color: 'text.secondary', fontWeight: 'bold' }}>
          External Maps
        </Typography>
        <List dense>
          <ListItem disablePadding>
            <ListItemButton component="a" href="https://egp.wildfire.gov/maps/?lat=34.877830&lon=-105.095146&zoom=4&dimension=2d" target="_blank">
              <ListItemIcon>
                <MdLocalFireDepartment size={24} color="#ff3300ff" />
              </ListItemIcon>
              <ListItemText primary="EGP Wildfire Maps" />
            </ListItemButton>
          </ListItem>
          <ListItem disablePadding>
            <ListItemButton component="a" href="https://www.weather.gov/fire/" target="_blank">
              <ListItemIcon>
                <MdWbSunny size={24} color="#e9ed02ff" />
              </ListItemIcon>
              <ListItemText primary="NWS Fire Weather" />
            </ListItemButton>
          </ListItem>
          <ListItem disablePadding>
            <ListItemButton component="a" href="https://www.cpc.ncep.noaa.gov/products/Drought/" target="_blank">
              <ListItemIcon>
                <MdWaterDrop size={24} color="#0288d1" />
              </ListItemIcon>
              <ListItemText primary="NOAA Drought Monitor" />
            </ListItemButton>
          </ListItem>
        </List>
      </Card>
      </Grid>
      <Dialog open={openWarningDialog} onClose={() => setOpenWarningDialog(false)}>
        <DialogTitle>Issue Fire Warning</DialogTitle>
        <DialogContent>
            <Typography gutterBottom>
                Select the warning radius (km) around high-risk stations:
            </Typography>
            <Box sx={{ px: 2, py: 1 }}>
                <Slider
                    value={warningRadius}
                    onChange={(e, newValue) => setWarningRadius(newValue)}
                    valueLabelDisplay="auto"
                    step={5}
                    marks
                    min={5}
                    max={100}
                />
            </Box>
            <Typography variant="body2" color="text.secondary">
                Radius: {warningRadius} km
            </Typography>
        </DialogContent>
        <DialogActions>
            <Button onClick={() => setOpenWarningDialog(false)}>Cancel</Button>
            <Button onClick={handleIssueWarning} variant="contained" color="error">
                Confirm Issue
            </Button>
        </DialogActions>
      </Dialog>

    </Grid>
  );
}

export default Overview;
