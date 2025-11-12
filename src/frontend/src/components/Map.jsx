import React, { useEffect } from 'react';
import { MapContainer, TileLayer, useMap, LayersControl } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import './Map.css';

const LIGHT_URL = 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png';
const SATELLITE_URL = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}';

function InvalidateOnResize({ wide }) {
  const map = useMap();
  useEffect(() => {
    setTimeout(() => {
      map.invalidateSize();
    }, 120); // allow CSS transition/layout settle
  }, [wide, map]);
  return null;
}

function Map({ wide }) {
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
      </MapContainer>
    </div>
  );
}

export default Map;