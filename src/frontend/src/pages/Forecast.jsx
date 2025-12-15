import React, { useEffect, useState } from 'react';
import { Grid, Button, Stack, Dialog, DialogTitle, DialogContent, DialogActions, Typography, TextField, MenuItem, FormControl, InputLabel, Select, Box, LinearProgress, Divider } from '@mui/material';
import InteractiveMap from '../components/InteractiveMap';
import Card from '../components/Card';
import { useForecast } from '../contexts/ForecastContext';

function Forecast() {
  const { forecastData, loading, fetchForecasts } = useForecast();
  
  const [wideMap, setWideMap] = useState(true);
  const [openAutoDialog, setOpenAutoDialog] = useState(false);
  const [openManualDialog, setOpenManualDialog] = useState(false);
  const [selectedStation, setSelectedStation] = useState('');
  const [resources, setResources] = useState({ personnel: '', equipment: '' });

  useEffect(() => {
    fetchForecasts();
  }, [fetchForecasts]);

  const handleAutoAllocate = () => {
    console.log("Allocating automatically...");
    setOpenAutoDialog(false);
  };

  const handleManualAllocate = () => {
    console.log(`Allocating manually to ${selectedStation}:`, resources);
    setOpenManualDialog(false);
    setSelectedStation('');
    setResources({ personnel: '', equipment: '' });
  };

  // Transform forecastData for InteractiveMap
  // forecastData: [{ name, location, forecast: [{ date, risk_level, ... }] }]
  // InteractiveMap expects: [{ name, lat, lon, data: { [offset]: { risk_level, ... } } }]
  
  const mapData = forecastData.map(station => {
    const dataMap = {};
    if (station.forecast) {
      station.forecast.forEach((day, index) => {
        // Calculate offset from today
        dataMap[index] = day;
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
      <Grid item xs={12} md={wideMap ? 8 : 6} sx={{ flex: { xs: '1 0 100%', md: wideMap ? 2 : 1 }, maxWidth: 'none' }}>
        <InteractiveMap 
          title="Forecast - Fire Weather Index"
          stationsData={mapData}
          timeRange={{ start: 0, end: 7 }}
          wide={wideMap}
          onToggleWide={() => setWideMap(w => !w)}
          loading={loading}
          referenceDate={new Date('2014-07-15')}
        />
      </Grid>
      <Grid item xs={12} md={wideMap ? 4 : 6} sx={{ flex: { xs: '1 0 100%', md: 1 }, maxWidth: 'none' }}>
        <Card title="Resource Allocation">
          <Box sx={{ p: 2 }}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>Current Utilization</Typography>
            
            <Box sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                    <Typography variant="body2">Personnel</Typography>
                    <Typography variant="body2" fontWeight="bold">78%</Typography>
                </Box>
                <LinearProgress variant="determinate" value={78} color="primary" sx={{ height: 8, borderRadius: 4 }} />
                <Typography variant="caption" color="text.secondary">94 / 120 deployed</Typography>
            </Box>

            <Box sx={{ mb: 3 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                    <Typography variant="body2">Equipment</Typography>
                    <Typography variant="body2" fontWeight="bold">65%</Typography>
                </Box>
                <LinearProgress variant="determinate" value={65} color="warning" sx={{ height: 8, borderRadius: 4 }} />
                <Typography variant="caption" color="text.secondary">32 / 50 units active</Typography>
            </Box>

            <Divider sx={{ mb: 2 }} />

            <Stack direction="column" spacing={2}>
                <Button variant="contained" color="primary" onClick={() => setOpenAutoDialog(true)}>
                Automatically reallocate
                </Button>
                <Button variant="outlined" color="primary" onClick={() => setOpenManualDialog(true)}>
                Allocate manually
                </Button>
            </Stack>
          </Box>
        </Card>
      </Grid>

      {/* Auto Allocation Dialog */}
      <Dialog open={openAutoDialog} onClose={() => setOpenAutoDialog(false)}>
        <DialogTitle>Automatic Resource Allocation</DialogTitle>
        <DialogContent>
          <Typography>
            Allocating resources automatically will initiate the process of moving resources based on the predicted risk levels.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenAutoDialog(false)}>Cancel</Button>
          <Button onClick={handleAutoAllocate} variant="contained" color="primary">
            Reallocate
          </Button>
        </DialogActions>
      </Dialog>

      {/* Manual Allocation Dialog */}
      <Dialog open={openManualDialog} onClose={() => setOpenManualDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Manual Resource Allocation</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
            <FormControl fullWidth>
              <InputLabel id="station-select-label">Select Station</InputLabel>
              <Select
                labelId="station-select-label"
                value={selectedStation}
                label="Select Station"
                onChange={(e) => setSelectedStation(e.target.value)}
              >
                {forecastData.map((station) => (
                  <MenuItem key={station.name} value={station.name}>
                    {station.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            
            <TextField
              label="Personnel"
              type="number"
              value={resources.personnel}
              onChange={(e) => setResources({ ...resources, personnel: e.target.value })}
              fullWidth
            />
            
            <TextField
              label="Equipment (Units)"
              type="number"
              value={resources.equipment}
              onChange={(e) => setResources({ ...resources, equipment: e.target.value })}
              fullWidth
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenManualDialog(false)}>Cancel</Button>
          <Button 
            onClick={handleManualAllocate} 
            variant="contained" 
            color="primary"
            disabled={!selectedStation || !resources.personnel || !resources.equipment}
          >
            Allocate
          </Button>
        </DialogActions>
      </Dialog>
    </Grid>
  );
}

export default Forecast;
