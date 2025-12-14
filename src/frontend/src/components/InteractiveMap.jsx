import React, { useState } from 'react';
import { Box, Typography, Slider, CircularProgress } from '@mui/material';
import Card from './Card';
import Map from './Map';
import './Content.css';

function InteractiveMap({ 
  title, 
  stationsData = [], 
  timeRange = { start: 0, end: 7 }, 
  defaultWide = false,
  wide: controlledWide,
  onToggleWide,
  loading = false,
  referenceDate = new Date()
}) {
  const [internalWide, setInternalWide] = useState(defaultWide);
  const [selectedDayOffset, setSelectedDayOffset] = useState(timeRange.start);

  const isControlled = controlledWide !== undefined;
  const wide = isControlled ? controlledWide : internalWide;

  const toggleWide = () => {
    if (onToggleWide) {
      onToggleWide();
    } else {
      setInternalWide(w => !w);
    }
  };

  const handleSliderChange = (event, newValue) => {
    setSelectedDayOffset(newValue);
  };

  // Calculate date label
  const getDateLabel = (offset) => {
    const date = new Date(referenceDate);
    date.setDate(date.getDate() + offset);
    return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
  };
  
  return (
    <Card
      title={title}
      className="card--bleed"
      actions={(
        <button
          type="button"
          className="map-expand-btn"
          aria-label={wide ? 'Collapse map width' : 'Expand map width'}
          onClick={toggleWide}
        >
          {wide ? 'Collapse' : 'Expand'}
        </button>
      )}
    >
      <div className={`map-slot ${wide ? 'map-slot--wide' : ''}`} style={{ height: '400px', position: 'relative' }}>
        {loading && (
          <Box
            sx={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              backgroundColor: 'rgba(255, 255, 255, 0.7)',
              zIndex: 1000,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <CircularProgress />
            <Typography variant="body2" sx={{ mt: 2 }}>
              Fetching predictions ...
            </Typography>
          </Box>
        )}
        <Map wide={wide} points={
            stationsData.map(station => {
                const dayData = station.data ? station.data[selectedDayOffset] : null;
                if (!dayData) return null;
                
                return {
                    lat: station.lat,
                    lon: station.lon,
                    risk_level: dayData.risk_level,
                    ...dayData
                };
            }).filter(Boolean)
        } />
      </div>
      <Box sx={{ px: 3, py: 2 }}>
        <Typography gutterBottom>
          {selectedDayOffset === 0 ? "Today" : 
           selectedDayOffset > 0 ? `Forecast: +${selectedDayOffset} days` : 
           `History: ${selectedDayOffset} days`} 
          <strong> ({getDateLabel(selectedDayOffset)})</strong>
        </Typography>
        <Slider
          value={selectedDayOffset}
          onChange={handleSliderChange}
          step={1}
          marks
          min={timeRange.start}
          max={timeRange.end}
          valueLabelDisplay="auto"
          valueLabelFormat={(value) => value > 0 ? `+${value}d` : `${value}d`}
        />
      </Box>
    </Card>
  );
}

export default InteractiveMap;
