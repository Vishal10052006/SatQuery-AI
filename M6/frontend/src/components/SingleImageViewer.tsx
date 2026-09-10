import { useState } from 'react';
import { Layers, ZoomIn, ZoomOut, RotateCcw, MapPin, Image as ImageIcon } from 'lucide-react';
import type { EvidenceItem, GisCoordinates, GisBoundingBox, GisGeoJsonFeature, SensorMetadata } from '../types';
import { GisMapViewer } from './GisMapViewer';

interface SingleImageViewerProps {
  imageUrl: string;
  overlayUrl?: string;
  title?: string;
  evidence?: EvidenceItem[];
  coordinates?: GisCoordinates;
  boundingBox?: GisBoundingBox;
  geojson?: GisGeoJsonFeature | GisGeoJsonFeature[];
  sensorMetadata?: SensorMetadata;
}

export const SingleImageViewer: React.FC<SingleImageViewerProps> = ({
  imageUrl,
  overlayUrl,
  title = 'Satellite Orthoimagery',
  coordinates,
  boundingBox,
  geojson,
  sensorMetadata,
}) => {
  const [activeTab, setActiveTab] = useState<'imagery' | 'gis-map'>('imagery');
  const [showOverlay, setShowOverlay] = useState(true);
  const [zoomLevel, setZoomLevel] = useState(1);

  const handleZoomIn = () => setZoomLevel((z) => Math.min(z + 0.25, 2.5));
  const handleZoomOut = () => setZoomLevel((z) => Math.max(z - 0.25, 0.75));
  const handleResetZoom = () => setZoomLevel(1);

  return (
    <div className="image-panel-card" style={{ marginBottom: 20 }}>
      <div className="image-panel-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontSize: '0.88rem', fontWeight: 600 }}>{title}</span>
          
          {/* Tab Switcher: Imagery vs GIS Map */}
          <div style={{ display: 'flex', background: 'var(--bg-input)', borderRadius: 'var(--radius-full)', padding: 2, border: '1px solid var(--border-subtle)' }}>
            <button
              type="button"
              className={`btn btn-sm ${activeTab === 'imagery' ? 'btn-primary' : ''}`}
              style={{ padding: '3px 10px', fontSize: '0.74rem', borderRadius: 'var(--radius-full)' }}
              onClick={() => setActiveTab('imagery')}
            >
              <ImageIcon size={13} /> Satellite Scene
            </button>
            <button
              type="button"
              className={`btn btn-sm ${activeTab === 'gis-map' ? 'btn-primary' : ''}`}
              style={{ padding: '3px 10px', fontSize: '0.74rem', borderRadius: 'var(--radius-full)' }}
              onClick={() => setActiveTab('gis-map')}
            >
              <MapPin size={13} /> GIS Footprint
            </button>
          </div>

          {activeTab === 'imagery' && overlayUrl && (
            <span className="badge badge-cyan" style={{ fontSize: '0.7rem' }}>
              Segmentation Ready
            </span>
          )}
        </div>

        {activeTab === 'imagery' && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {overlayUrl && (
              <button
                type="button"
                className={`btn btn-sm ${showOverlay ? 'btn-primary' : 'btn-secondary'}`}
                onClick={() => setShowOverlay(!showOverlay)}
                title="Toggle AI feature classification layer"
              >
                <Layers size={13} />
                {showOverlay ? 'Hide Classification Layer' : 'Show Classification Layer'}
              </button>
            )}

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
                {Math.round(zoomLevel * 100)}%
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
                onClick={handleResetZoom}
                style={{ background: 'transparent', padding: '4px 8px', borderLeft: '1px solid var(--border-subtle)' }}
                title="Reset Zoom"
              >
                <RotateCcw size={12} />
              </button>
            </div>
          </div>
        )}
      </div>

      {activeTab === 'gis-map' ? (
        <GisMapViewer
          coordinates={coordinates || { lat: 18.975, lng: 72.825, zoom: 12, locationName: 'Mumbai Coastal Estuary & Urban Harbor' }}
          boundingBox={boundingBox || { north: 19.08, south: 18.88, east: 72.96, west: 72.74 }}
          geojson={geojson}
          sensorMetadata={sensorMetadata}
          title="Satellite Scene GIS Footprint & Bounding Reticle"
        />
      ) : (
        <div
          className="image-panel-body"
          style={{
            height: '460px',
            overflow: 'hidden',
            position: 'relative',
            cursor: zoomLevel > 1 ? 'grab' : 'default',
          }}
        >
          <div
            style={{
              position: 'relative',
              width: '100%',
              height: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transform: `scale(${zoomLevel})`,
              transition: 'transform 0.2s ease-out',
            }}
          >
            {/* Base Satellite Image */}
            <img
              src={imageUrl}
              alt="Satellite Scene"
              style={{
                maxWidth: '100%',
                maxHeight: '100%',
                objectFit: 'contain',
                display: 'block',
              }}
            />

            {/* Optional Classification / Thematic Mask Overlay */}
            {showOverlay && overlayUrl && (
              <img
                src={overlayUrl}
                alt="Feature Segmentation Mask"
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: '100%',
                  objectFit: 'contain',
                  pointerEvents: 'none',
                }}
              />
            )}
          </div>

          {/* Top HUD Telemetry Watermark */}
          <div
            style={{
              position: 'absolute',
              top: 14,
              left: 14,
              background: 'rgba(8, 12, 22, 0.85)',
              border: '1px solid var(--border-medium)',
              padding: '4px 10px',
              borderRadius: 'var(--radius-sm)',
              fontSize: '0.72rem',
              fontFamily: 'var(--font-mono)',
              color: 'var(--cyan-bright)',
              backdropFilter: 'blur(6px)',
              zIndex: 5,
            }}
          >
            LOC: {coordinates ? `${coordinates.lat.toFixed(2)}°N ${coordinates.lng.toFixed(2)}°E` : '18.98°N 72.83°E'} • SENTINEL-2 L2A • GSD: 10M
          </div>

          {/* Informational overlay tags */}
          <div
            style={{
              position: 'absolute',
              bottom: 12,
              left: 12,
              display: 'flex',
              gap: 8,
              zIndex: 10,
            }}
          >
            <span
              style={{
                background: 'rgba(10, 15, 29, 0.85)',
                border: '1px solid var(--border-medium)',
                padding: '3px 8px',
                borderRadius: 'var(--radius-sm)',
                fontSize: '0.72rem',
                color: 'var(--text-secondary)',
                backdropFilter: 'blur(4px)',
              }}
            >
              Base: {sensorMetadata?.platform || 'Sentinel-2 MSI Multispectral'}
            </span>
            {showOverlay && overlayUrl && (
              <span
                style={{
                  background: 'rgba(6, 182, 212, 0.2)',
                  border: '1px solid var(--cyan)',
                  padding: '3px 8px',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: '0.72rem',
                  color: 'var(--cyan)',
                  backdropFilter: 'blur(4px)',
                  fontWeight: 600,
                }}
              >
                Thematic Classification Layer Active
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
