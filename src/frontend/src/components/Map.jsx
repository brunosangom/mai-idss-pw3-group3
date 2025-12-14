import React, { useEffect } from 'react';
import { MapContainer, TileLayer, useMap, LayersControl, CircleMarker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import './Map.css';

const LIGHT_URL = 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png';
const SATELLITE_URL = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}';

function InvalidateOnResize({ wide }) {
  const map = useMap();
  useEffect(() => {
    const timer = setTimeout(() => {
      if (map) {
        try {
          map.invalidateSize();
        } catch (e) {
          console.warn("Map invalidateSize failed", e);
        }
      }
    }, 120); // allow CSS transition/layout settle
    return () => clearTimeout(timer);
  }, [wide, map]);
  return null;
}

const getRiskColor = (level) => {
  switch (level) {
    case 'Low': return '#4caf50'; // Green
    case 'Moderate': return '#ffeb3b'; // Yellow
    case 'High': return '#ff9800'; // Orange
    case 'Extreme': return '#f44336'; // Red
    case 'Very High': return '#ff5722'; // Deep Orange
    default: return '#9e9e9e'; // Grey
  }
};

function Map({ wide, points = [] }) {
  return (
    <div className="map-container">
      <MapContainer center={[37.0902, -95.7129]} zoom={4} scrollWheelZoom={true} className="map">
        <InvalidateOnResize wide={wide} />
        <LayersControl position="topright">
          <LayersControl.BaseLayer checked name="Light">
            <TileLayer
              url={LIGHT_URL}
              subdomains={["a","b","c","d"]}
              attribution="&copy; OpenStreetMap contributors &copy; CARTO"
            />
          </LayersControl.BaseLayer>
          <LayersControl.BaseLayer name="Satellite">
            <TileLayer
              url={SATELLITE_URL}
              attribution="Imagery &copy; Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community"
            />
          </LayersControl.BaseLayer>
        </LayersControl>

        {points.map((point, index) => {
          if (!point) return null;

          return (
            <CircleMarker
              key={`${point.lat}-${point.lon}-${index}`}
              center={[point.lat, point.lon]}
              radius={10}
              pathOptions={{
                color: 'white',
                weight: 1,
                fillColor: getRiskColor(point.risk_level),
                fillOpacity: 0.8
              }}
            >
              <Popup>
                <div style={{ textAlign: 'center' }}>
                  <strong>{point.name || 'Unknown Location'}</strong><br/>
                  Date: {point.date}<br/>
                  Risk: <span style={{ color: getRiskColor(point.risk_level), fontWeight: 'bold' }}>
                    {point.risk_level}
                  </span><br/>
                  {point.fwi_value !== undefined && <>FWI: {point.fwi_value}</>}
                </div>
              </Popup>
            </CircleMarker>
          );
        })}
      </MapContainer>
    </div>
  );
}

export default Map;