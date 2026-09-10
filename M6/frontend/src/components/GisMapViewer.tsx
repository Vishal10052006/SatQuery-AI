import React, { useState } from 'react';
import { 
  MapPin, 
  ZoomIn, 
  ZoomOut, 
  RotateCcw, 
  Compass, 
  Maximize2, 
  Minimize2,
  Navigation
} from 'lucide-react';
import type { GisCoordinates, GisBoundingBox, GisGeoJsonFeature, SensorMetadata } from '../types';

interface GisMapViewerProps {
  coordinates?: GisCoordinates;
  boundingBox?: GisBoundingBox;
  geojson?: GisGeoJsonFeature | GisGeoJsonFeature[];
  sensorMetadata?: SensorMetadata;
  title?: string;
  isDemoLocation?: boolean;
}

export const GisMapViewer: React.FC<GisMapViewerProps> = ({
  coordinates = { lat: 18.975, lng: 72.825, zoom: 12, locationName: 'Mumbai Estuary & Harbor Basin', crs: 'EPSG:4326' },
  boundingBox = { north: 19.08, south: 18.88, east: 72.96, west: 72.74 },
  geojson,
  sensorMetadata = {
    platform: 'Sentinel-2A MSI',
    sensor: 'MultiSpectral Level-2A',
    gsd: '10m / px',
    crs: 'EPSG:4326 (WGS 84)',
  },
  title = 'Geospatial GIS Context & Coordinate Mapping',
  isDemoLocation = false,
}) => {
  const [zoom, setZoom] = useState(coordinates.zoom || 12);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [startPan, setStartPan] = useState({ x: 0, y: 0 });
  const [mapStyle, setMapStyle] = useState<'carto-dark' | 'satellite' | 'vector-grid'>('carto-dark');
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [selectedFeature, setSelectedFeature] = useState<string | null>(null);

  const featuresList = Array.isArray(geojson) ? geojson : geojson ? [geojson] : [];

  const handleZoomIn = () => setZoom((z) => Math.min(z + 1, 18));
  const handleZoomOut = () => setZoom((z) => Math.max(z - 1, 6));
  const handleReset = () => {
    setZoom(coordinates.zoom || 12);
    setPanOffset({ x: 0, y: 0 });
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsPanning(true);
    setStartPan({ x: e.clientX - panOffset.x, y: e.clientY - panOffset.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isPanning) return;
    setPanOffset({
      x: e.clientX - startPan.x,
      y: e.clientY - startPan.y,
    });
  };

  const handleMouseUp = () => setIsPanning(false);

  const latFormatted = `${Math.abs(coordinates.lat).toFixed(4)}°${coordinates.lat >= 0 ? 'N' : 'S'}`;
  const lngFormatted = `${Math.abs(coordinates.lng).toFixed(4)}°${coordinates.lng >= 0 ? 'E' : 'W'}`;

  return (
    <div
      className={`image-panel-card gis-map-card ${isFullscreen ? 'gis-fullscreen' : ''}`}
      style={{
        marginBottom: 20,
        position: isFullscreen ? 'fixed' : 'relative',
        top: isFullscreen ? 0 : 'auto',
        left: isFullscreen ? 0 : 'auto',
        width: isFullscreen ? '100vw' : '100%',
        height: isFullscreen ? '100vh' : 'auto',
        zIndex: isFullscreen ? 9999 : 'auto',
        borderRadius: isFullscreen ? 0 : undefined,
      }}
    >
      {/* Top Header Bar */}
      <div className="image-panel-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <Compass size={18} color="var(--cyan)" />
          <div>
            <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>{title}</span>
            {isDemoLocation && (
              <span className="badge badge-cyan" style={{ marginLeft: 8, fontSize: '0.68rem' }}>
                DEMO GIS LAYER
              </span>
            )}
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          {/* Map Style Selector */}
          <div style={{ display: 'flex', background: 'var(--bg-input)', borderRadius: 'var(--radius-full)', padding: 2, border: '1px solid var(--border-subtle)' }}>
            <button
              type="button"
              className={`btn btn-sm ${mapStyle === 'carto-dark' ? 'btn-primary' : ''}`}
              style={{ padding: '3px 10px', fontSize: '0.74rem', borderRadius: 'var(--radius-full)' }}
              onClick={() => setMapStyle('carto-dark')}
            >
              Dark Matter
            </button>
            <button
              type="button"
              className={`btn btn-sm ${mapStyle === 'satellite' ? 'btn-primary' : ''}`}
              style={{ padding: '3px 10px', fontSize: '0.74rem', borderRadius: 'var(--radius-full)' }}
              onClick={() => setMapStyle('satellite')}
            >
              Satellite Hybrid
            </button>
            <button
              type="button"
              className={`btn btn-sm ${mapStyle === 'vector-grid' ? 'btn-primary' : ''}`}
              style={{ padding: '3px 10px', fontSize: '0.74rem', borderRadius: 'var(--radius-full)' }}
              onClick={() => setMapStyle('vector-grid')}
            >
              Grid HUD
            </button>
          </div>

          {/* Zoom & Fullscreen Controls */}
          <div style={{ display: 'flex', alignItems: 'center', background: 'var(--bg-input)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
            <button
              type="button"
              className="btn btn-sm"
              onClick={handleZoomOut}
              style={{ background: 'transparent', padding: '4px 8px' }}
              title="Zoom Out"
            >
              <ZoomOut size={13} />
            </button>
            <span style={{ fontSize: '0.72rem', fontFamily: 'var(--font-mono)', padding: '0 4px', color: 'var(--text-muted)' }}>
              z{zoom}
            </span>
            <button
              type="button"
              className="btn btn-sm"
              onClick={handleZoomIn}
              style={{ background: 'transparent', padding: '4px 8px' }}
              title="Zoom In"
            >
              <ZoomIn size={13} />
            </button>
            <button
              type="button"
              className="btn btn-sm"
              onClick={handleReset}
              style={{ background: 'transparent', padding: '4px 8px', borderLeft: '1px solid var(--border-subtle)' }}
              title="Reset View"
            >
              <RotateCcw size={12} />
            </button>
          </div>

          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={() => setIsFullscreen(!isFullscreen)}
            title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen GIS Viewer'}
            style={{ padding: '5px 8px' }}
          >
            {isFullscreen ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
          </button>
        </div>
      </div>

      {/* Interactive Map Canvas Container */}
      <div
        className="gis-canvas-wrapper"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{
          height: isFullscreen ? 'calc(100vh - 120px)' : '460px',
          overflow: 'hidden',
          position: 'relative',
          cursor: isPanning ? 'grabbing' : 'grab',
          userSelect: 'none',
          backgroundColor: '#040711',
        }}
      >
        {/* Dynamic Spatial Grid & Vector Graphic Layer */}
        <div
          style={{
            position: 'absolute',
            width: '100%',
            height: '100%',
            transform: `translate(${panOffset.x}px, ${panOffset.y}px) scale(${1 + (zoom - 10) * 0.12})`,
            transformOrigin: 'center center',
            transition: isPanning ? 'none' : 'transform 0.15s ease-out',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          {/* Background Map Simulation */}
          <div
            className={`gis-basemap-layer ${mapStyle}`}
            style={{
              width: '960px',
              height: '640px',
              position: 'relative',
              borderRadius: 8,
              boxShadow: '0 0 40px rgba(0, 0, 0, 0.8)',
              overflow: 'hidden',
            }}
          >
            {/* SVG GIS Topographic Grid and Coordinate Matrix */}
            <svg width="100%" height="100%" viewBox="0 0 960 640" style={{ position: 'absolute', top: 0, left: 0 }}>
              <defs>
                <pattern id="gis-grid-pattern" width="48" height="48" patternUnits="userSpaceOnUse">
                  <path d="M 48 0 L 0 0 0 48" fill="none" stroke="rgba(6, 182, 212, 0.12)" strokeWidth="1" />
                  <circle cx="0" cy="0" r="1.5" fill="rgba(6, 182, 212, 0.4)" />
                </pattern>
                <linearGradient id="bbox-glow" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="rgba(6, 182, 212, 0.35)" />
                  <stop offset="100%" stopColor="rgba(99, 102, 241, 0.35)" />
                </linearGradient>
              </defs>

              {/* Grid Background */}
              <rect width="960" height="640" fill="url(#gis-grid-pattern)" />

              {/* Major Coordinate Graticule Lines */}
              <line x1="160" y1="0" x2="160" y2="640" stroke="rgba(255, 255, 255, 0.08)" strokeDasharray="4,4" />
              <line x1="480" y1="0" x2="480" y2="640" stroke="rgba(6, 182, 212, 0.25)" strokeDasharray="6,4" />
              <line x1="800" y1="0" x2="800" y2="640" stroke="rgba(255, 255, 255, 0.08)" strokeDasharray="4,4" />
              <line x1="0" y1="320" x2="960" y2="320" stroke="rgba(6, 182, 212, 0.25)" strokeDasharray="6,4" />

              {/* Coastline / Topographic Contour Simulator */}
              <path
                d="M 60,640 Q 240,480 320,320 T 560,180 Q 720,120 960,80 L 960,640 Z"
                fill="rgba(15, 23, 42, 0.65)"
                stroke="rgba(6, 182, 212, 0.3)"
                strokeWidth="2"
              />

              {/* Satellite Footprint Bounding Box */}
              <rect
                x="220"
                y="120"
                width="520"
                height="380"
                fill="url(#bbox-glow)"
                stroke="var(--cyan)"
                strokeWidth="2"
                strokeDasharray="8,5"
                rx="6"
              />

              {/* Bounding Box Corner Reticles */}
              <path d="M 215,135 L 215,115 L 235,115" fill="none" stroke="var(--cyan-bright)" strokeWidth="3" />
              <path d="M 745,135 L 745,115 L 725,115" fill="none" stroke="var(--cyan-bright)" strokeWidth="3" />
              <path d="M 215,485 L 215,505 L 235,505" fill="none" stroke="var(--cyan-bright)" strokeWidth="3" />
              <path d="M 745,485 L 745,505 L 725,505" fill="none" stroke="var(--cyan-bright)" strokeWidth="3" />

              {/* GeoJSON Feature Overlays (Polygons / Points) */}
              {featuresList.map((feat, idx) => {
                const color = feat.properties?.color || 'var(--cyan)';
                if (feat.geometry.type === 'Point') {
                  const cx = 320 + (idx * 90) % 360;
                  const cy = 200 + (idx * 65) % 240;
                  return (
                    <g key={idx} onClick={() => setSelectedFeature(feat.properties?.name || `Target ${idx + 1}`)} style={{ cursor: 'pointer' }}>
                      <circle cx={cx} cy={cy} r="18" fill="none" stroke={color} strokeWidth="1.5" strokeDasharray="3,3" />
                      <circle cx={cx} cy={cy} r="6" fill={color} stroke="#050811" strokeWidth="2" />
                      <text x={cx + 12} y={cy + 4} fill="#f8fafc" fontSize="11" fontFamily="sans-serif" fontWeight="bold">
                        {feat.properties?.name || `Target #${idx + 1}`}
                      </text>
                    </g>
                  );
                }
                return (
                  <g key={idx} onClick={() => setSelectedFeature(feat.properties?.name || `Zone ${idx + 1}`)} style={{ cursor: 'pointer' }}>
                    <polygon
                      points={idx === 0 ? "260,160 520,160 540,320 280,300" : "540,240 700,240 710,460 530,440"}
                      fill={color}
                      fillOpacity="0.32"
                      stroke={color}
                      strokeWidth="2.5"
                    />
                    <text x={idx === 0 ? 300 : 560} y={idx === 0 ? 230 : 340} fill="#ffffff" fontSize="12" fontWeight="bold" filter="drop-shadow(0 2px 4px #000)">
                      {feat.properties?.name || `Feature Layer #${idx + 1}`}
                    </text>
                  </g>
                );
              })}

              {/* Centroid Crosshair Marker */}
              <g transform="translate(480, 310)">
                <circle r="28" fill="none" stroke="var(--cyan-bright)" strokeWidth="1" strokeDasharray="4,2" opacity="0.8" />
                <circle r="12" fill="none" stroke="var(--cyan-bright)" strokeWidth="2" />
                <line x1="-34" y1="0" x2="34" y2="0" stroke="var(--cyan-bright)" strokeWidth="1.5" />
                <line x1="0" y1="-34" x2="0" y2="34" stroke="var(--cyan-bright)" strokeWidth="1.5" />
                <circle r="3" fill="#ffffff" />
              </g>

              {/* Bounding Coordinates Labels */}
              <text x="480" y="110" textAnchor="middle" fill="var(--cyan-bright)" fontSize="11" fontFamily="monospace">
                NORTH: {boundingBox.north.toFixed(4)}°N
              </text>
              <text x="480" y="525" textAnchor="middle" fill="var(--cyan-bright)" fontSize="11" fontFamily="monospace">
                SOUTH: {boundingBox.south.toFixed(4)}°S
              </text>
              <text x="210" y="310" textAnchor="end" fill="var(--cyan-bright)" fontSize="11" fontFamily="monospace" transform="rotate(-90 210,310)">
                WEST: {boundingBox.west.toFixed(4)}°W
              </text>
              <text x="755" y="310" textAnchor="start" fill="var(--cyan-bright)" fontSize="11" fontFamily="monospace" transform="rotate(90 755,310)">
                EAST: {boundingBox.east.toFixed(4)}°E
              </text>
            </svg>
          </div>
        </div>

        {/* Top-Left Geographic Location HUD */}
        <div
          style={{
            position: 'absolute',
            top: 14,
            left: 14,
            background: 'rgba(8, 12, 22, 0.88)',
            border: '1px solid var(--border-medium)',
            padding: '8px 14px',
            borderRadius: 'var(--radius-sm)',
            fontSize: '0.75rem',
            backdropFilter: 'blur(8px)',
            zIndex: 10,
            maxWidth: 380,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 2 }}>
            <MapPin size={14} color="var(--cyan)" />
            {coordinates.locationName || 'Satellite Ortho Scene Footprint'}
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem', color: 'var(--cyan-bright)' }}>
            CENTER: {latFormatted}, {lngFormatted} • CRS: {coordinates.crs || 'EPSG:4326'}
          </div>
        </div>

        {/* Top-Right Telemetry Badge */}
        <div
          style={{
            position: 'absolute',
            top: 14,
            right: 14,
            background: 'rgba(8, 12, 22, 0.88)',
            border: '1px solid var(--border-medium)',
            padding: '8px 12px',
            borderRadius: 'var(--radius-sm)',
            fontSize: '0.72rem',
            fontFamily: 'var(--font-mono)',
            backdropFilter: 'blur(8px)',
            zIndex: 10,
            textAlign: 'right',
          }}
        >
          <div style={{ color: 'var(--emerald-bright)' }}>PLATFORM: {sensorMetadata.platform}</div>
          <div style={{ color: 'var(--text-muted)' }}>SENSOR: {sensorMetadata.sensor}</div>
          <div style={{ color: 'var(--text-muted)' }}>RESOLUTION: {sensorMetadata.gsd}</div>
        </div>

        {/* Selected Feature Inspector Callout */}
        {selectedFeature && (
          <div
            style={{
              position: 'absolute',
              bottom: 48,
              left: 14,
              background: 'rgba(10, 15, 29, 0.95)',
              border: '1px solid var(--cyan)',
              padding: '6px 12px',
              borderRadius: 'var(--radius-sm)',
              fontSize: '0.75rem',
              color: '#ffffff',
              backdropFilter: 'blur(8px)',
              zIndex: 10,
              display: 'flex',
              alignItems: 'center',
              gap: 8,
            }}
          >
            <Navigation size={14} color="var(--cyan)" />
            <span>Inspecting Feature: <strong>{selectedFeature}</strong></span>
            <button
              type="button"
              onClick={() => setSelectedFeature(null)}
              style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', marginLeft: 6 }}
            >
              ✕
            </button>
          </div>
        )}

        {/* Bottom Legend & Guidance Bar */}
        <div
          style={{
            position: 'absolute',
            bottom: 12,
            left: 14,
            right: 14,
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            zIndex: 10,
            pointerEvents: 'none',
          }}
        >
          <div style={{ display: 'flex', gap: 8, pointerEvents: 'auto' }}>
            <span style={{ background: 'rgba(8, 12, 22, 0.85)', padding: '4px 10px', borderRadius: 4, border: '1px solid var(--border-subtle)', fontSize: '0.7rem', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--cyan)' }} />
              Satellite Footprint (AOI)
            </span>
            <span style={{ background: 'rgba(8, 12, 22, 0.85)', padding: '4px 10px', borderRadius: 4, border: '1px solid var(--border-subtle)', fontSize: '0.7rem', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#f43f5e' }} />
              GeoJSON Vector Polygon
            </span>
          </div>

          <div style={{ background: 'rgba(8, 12, 22, 0.85)', padding: '4px 12px', borderRadius: 20, border: '1px solid var(--border-subtle)', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
            ↔ Drag to Pan • Buttons to Zoom
          </div>
        </div>
      </div>
    </div>
  );
};
