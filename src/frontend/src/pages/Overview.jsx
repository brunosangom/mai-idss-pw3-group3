import React, { useState } from 'react';
import { Grid, List, ListItem, ListItemButton, ListItemText, ListItemIcon, Divider, Typography } from '@mui/material';
import { MdLocalFireDepartment, MdWbSunny, MdWaterDrop, MdTerrain, MdInfo } from 'react-icons/md';
import '../components/Content.css';
import Card from '../components/Card';
import Map from '../components/Map';

function Overview() {
  const [wideMap, setWideMap] = useState(false);
  const toggleWide = () => setWideMap(w => !w);

  return (
    <Grid container spacing={2} sx={{ width: '100%' }}>
      <Grid item xs={12} md={wideMap ? 8 : 6} sx={{ flex: { xs: '1 0 100%', md: 2 }, maxWidth: 'none' }}>
        <Card
          title="Map"
          className="card--bleed"
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
      </Grid>

      <Grid item xs={12} md={wideMap ? 4 : 3} sx={{ flex: { xs: '1 0 100%', md: 1 }, maxWidth: 'none' }}>
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
    </Grid>
  );
}

export default Overview;
