import React, { useState, useRef, useEffect, useCallback } from 'react';
import { 
  SplitSquareVertical, 
  Columns, 
  Layers,
  MapPin,
  Image as ImageIcon
} from 'lucide-react';
import type { DetectedChange, GisCoordinates, GisBoundingBox, GisGeoJsonFeature, SensorMetadata } from '../types';
import { GisMapViewer } from './GisMapViewer';

interface ComparisonViewerProps {
  beforeUrl: string;
  afterUrl: string;
  overlayUrl?: string;
  changes?: DetectedChange[];
  beforeTitle?: string;
  afterTitle?: string;
  coordinates?: GisCoordinates;
  boundingBox?: GisBoundingBox;
  geojson?: GisGeoJsonFeature | GisGeoJsonFeature[];
  sensorMetadata?: SensorMetadata;
}

export const ComparisonViewer: React.FC<ComparisonViewerProps> = ({
  beforeUrl,
  afterUrl,
  overlayUrl,
  changes = [],
  beforeTitle = 'Timestamp T1 (Before)',
  afterTitle = 'Timestamp T2 (After)',
  coordinates,
  boundingBox,
  geojson,
  sensorMetadata,
}) => {
  const [activeTab, setActiveTab] = useState<'imagery' | 'gis-map'>('imagery');
  const [viewStyle, setViewStyle] = useState<'slider' | 'side-by-side'>('slider');
  const [sliderPos, setSliderPos] = useState(50); // percentage 0 - 100
  const [isDragging, setIsDragging] = useState(false);
  const [showOverlay, setShowOverlay] = useState(true);
  const [overlayOpacity, setOverlayOpacity] = useState(70); // 0 - 100%
  const [containerWidth, setContainerWidth] = useState<number | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const el = containerRef.current;
    const updateWidth = () => {
      setContainerWidth(el.clientWidth);
    };
    updateWidth();
    const observer = new ResizeObserver(updateWidth);
    observer.observe(el);
    return () => observer.disconnect();
  }, [viewStyle, activeTab]);

  const handlePointerDown = () => {
    setIsDragging(true);
  };

  const handlePointerUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handlePointerMove = useCallback((e: MouseEvent | TouchEvent) => {
    if (!isDragging || !containerRef.current) return;

    const rect = containerRef.current.getBoundingClientRect();
    const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
    const offsetX = clientX - rect.left;
    const percentage = Math.max(0, Math.min(100, (offsetX / rect.width) * 100));
    setSliderPos(percentage);
  }, [isDragging]);

  useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handlePointerMove);
      window.addEventListener('mouseup', handlePointerUp);
      window.addEventListener('touchmove', handlePointerMove);
      window.addEventListener('touchend', handlePointerUp);
    }
    return () => {
      window.removeEventListener('mousemove', handlePointerMove);
      window.removeEventListener('mouseup', handlePointerUp);
      window.removeEventListener('touchmove', handlePointerMove);
      window.removeEventListener('touchend', handlePointerUp);
    };
  }, [isDragging, handlePointerMove, handlePointerUp]);

  return (
    <div className="image-panel-card" style={{ marginBottom: 20 }}>
      {/* Top Toolbar */}
      <div className="image-panel-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>Bi-Temporal Change Inspector</span>

          {/* Tab: Satellite Imagery vs GIS Map */}
          <div style={{ display: 'flex', background: 'var(--bg-input)', borderRadius: 'var(--radius-full)', padding: 2, border: '1px solid var(--border-subtle)' }}>
            <button
              type="button"
              className={`btn btn-sm ${activeTab === 'imagery' ? 'btn-primary' : ''}`}
              style={{ padding: '3px 10px', fontSize: '0.75rem', borderRadius: 'var(--radius-full)' }}
              onClick={() => setActiveTab('imagery')}
            >
              <ImageIcon size={13} /> Satellite Swipe
            </button>
            <button
              type="button"
              className={`btn btn-sm ${activeTab === 'gis-map' ? 'btn-primary' : ''}`}
              style={{ padding: '3px 10px', fontSize: '0.75rem', borderRadius: 'var(--radius-full)' }}
              onClick={() => setActiveTab('gis-map')}
            >
              <MapPin size={13} /> GIS Map View
            </button>
          </div>
          
          {/* Switcher: Slider vs Side-by-Side (only in imagery mode) */}
          {activeTab === 'imagery' && (
            <div style={{ display: 'flex', background: 'var(--bg-input)', borderRadius: 'var(--radius-full)', padding: 2, border: '1px solid var(--border-subtle)' }}>
              <button
                type="button"
                className={`btn btn-sm ${viewStyle === 'slider' ? 'btn-primary' : ''}`}
                style={{ padding: '3px 10px', fontSize: '0.75rem', borderRadius: 'var(--radius-full)' }}
                onClick={() => setViewStyle('slider')}
              >
                <SplitSquareVertical size={13} /> Split Slider
              </button>
              <button
                type="button"
                className={`btn btn-sm ${viewStyle === 'side-by-side' ? 'btn-primary' : ''}`}
                style={{ padding: '3px 10px', fontSize: '0.75rem', borderRadius: 'var(--radius-full)' }}
                onClick={() => setViewStyle('side-by-side')}
              >
                <Columns size={13} /> Side-by-Side
              </button>
            </div>
          )}
        </div>

        {/* Change Overlay Controls */}
        {activeTab === 'imagery' && overlayUrl && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <button
              type="button"
              className={`btn btn-sm ${showOverlay ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setShowOverlay(!showOverlay)}
            >
              <Layers size={13} />
              {showOverlay ? 'Hide Change Mask' : 'Show Change Mask'}
            </button>

            {showOverlay && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                <span>Mask Opacity:</span>
                <input
                  type="range"
                  min="10"
                  max="100"
                  value={overlayOpacity}
                  onChange={(e) => setOverlayOpacity(Number(e.target.value))}
                  style={{ width: 80, accentColor: 'var(--cyan)' }}
                />
                <span style={{ fontFamily: 'var(--font-mono)', minWidth: 32 }}>{overlayOpacity}%</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Main View Area */}
      {activeTab === 'gis-map' ? (
        <GisMapViewer
          coordinates={coordinates || { lat: 26.182, lng: 91.745, zoom: 11, locationName: 'Brahmaputra River Basin Inundation Zone' }}
          boundingBox={boundingBox || { north: 26.32, south: 26.04, east: 91.95, west: 91.55 }}
          geojson={geojson}
          sensorMetadata={sensorMetadata}
          title="Georeferenced Change Boundary & Flood Inundation GIS Mapping"
        />
      ) : viewStyle === 'slider' ? (
        <div
          ref={containerRef}
          className="comparison-wrapper"
          onMouseDown={handlePointerDown}
          onTouchStart={handlePointerDown}
          style={{ cursor: 'ew-resize' }}
        >
          {/* Base Layer: AFTER Image */}
          <div className="comparison-layer">
            <img src={afterUrl} alt="After Scene (T2)" className="comparison-img" />
          </div>

          {/* Change Mask on top of AFTER Image if enabled */}
          {showOverlay && overlayUrl && (
            <div
              className="comparison-layer"
              style={{
                opacity: overlayOpacity / 100,
                pointerEvents: 'none',
                mixBlendMode: 'screen',
              }}
            >
              <img src={overlayUrl} alt="Change Detection Mask" className="comparison-img" />
            </div>
          )}

          {/* Clipped Layer: BEFORE Image */}
          <div
            className="comparison-overlay-clip"
            style={{ width: `${sliderPos}%` }}
          >
            <div style={{ width: containerWidth ? `${containerWidth}px` : '100%', height: '100%' }}>
              <img
                src={beforeUrl}
                alt="Before Scene (T1)"
                style={{
                  width: containerWidth ? `${containerWidth}px` : '100%',
                  height: '100%',
                  objectFit: 'cover',
                  display: 'block',
                }}
              />
            </div>
          </div>

          {/* Split Slider Draggable Divider Line and Handle */}
          <div
            className="comparison-slider-handle"
            style={{ left: `${sliderPos}%` }}
          >
            <SplitSquareVertical size={18} />
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
            LOC: {coordinates ? `${coordinates.lat.toFixed(2)}°N ${coordinates.lng.toFixed(2)}°E` : '26.18°N 91.75°E'} • GSD: 10M/PX • EPSG:4326
          </div>

          {/* Bottom Guidance Indicator */}
          <div
            style={{
              position: 'absolute',
              bottom: 18,
              left: '50%',
              transform: 'translateX(-50%)',
              background: 'rgba(8, 12, 22, 0.85)',
              border: '1px solid var(--border-medium)',
              padding: '4px 12px',
              borderRadius: 'var(--radius-full)',
              fontSize: '0.72rem',
              color: 'var(--text-secondary)',
              backdropFilter: 'blur(6px)',
              zIndex: 5,
              pointerEvents: 'none',
            }}
          >
            ↔ Drag cursor horizontally to inspect before/after changes
          </div>

          {/* Badges indicating which side is which */}
          <div className="comparison-label-badge left">
            ◀ {beforeTitle}
          </div>
          <div className="comparison-label-badge right">
            {afterTitle} ▶
          </div>
        </div>
      ) : (
        /* Side-by-Side Dual View */
        <div className="dual-viewer-grid" style={{ padding: 12 }}>
          {/* Before Panel */}
          <div style={{ background: '#020617', borderRadius: 'var(--radius-md)', overflow: 'hidden', border: '1px solid var(--border-subtle)', position: 'relative' }}>
            <div style={{ padding: '8px 12px', background: 'var(--bg-card-elevated)', fontSize: '0.78rem', fontWeight: 600, color: 'var(--cyan)' }}>
              {beforeTitle}
            </div>
            <div style={{ height: 380, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <img src={beforeUrl} alt="Before Scene" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
            </div>
          </div>

          {/* After Panel */}
          <div style={{ background: '#020617', borderRadius: 'var(--radius-md)', overflow: 'hidden', border: '1px solid var(--border-subtle)', position: 'relative' }}>
            <div style={{ padding: '8px 12px', background: 'var(--bg-card-elevated)', fontSize: '0.78rem', fontWeight: 600, color: 'var(--emerald)' }}>
              {afterTitle}
            </div>
            <div style={{ height: 380, display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
              <img src={afterUrl} alt="After Scene" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
              {showOverlay && overlayUrl && (
                <img
                  src={overlayUrl}
                  alt="Change Mask"
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    objectFit: 'cover',
                    opacity: overlayOpacity / 100,
                    pointerEvents: 'none',
                  }}
                />
              )}
            </div>
          </div>
        </div>
      )}

      {/* Quantitative change summary strip */}
      {changes && changes.length > 0 && (
        <div style={{ padding: '14px 18px', background: 'var(--bg-card-elevated)', borderTop: '1px solid var(--border-subtle)', display: 'flex', flexWrap: 'wrap', gap: 14 }}>
          {changes.map((ch) => (
            <div
              key={ch.id}
              style={{
                background: 'var(--bg-input)',
                border: '1px solid var(--border-medium)',
                borderRadius: 'var(--radius-md)',
                padding: '8px 14px',
                flex: '1 1 240px',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
                <span style={{ fontSize: '0.82rem', fontWeight: 600, color: ch.severity === 'high' ? 'var(--rose)' : 'var(--amber)' }}>
                  {ch.type}
                </span>
                <span className="badge badge-amber" style={{ fontSize: '0.7rem' }}>
                  {ch.estimatedAreaHa} Ha
                </span>
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                {ch.description}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
