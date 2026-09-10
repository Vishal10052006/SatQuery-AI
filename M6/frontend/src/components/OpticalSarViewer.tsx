import { useState } from 'react';
import { Layers, Radio, Sun, MapPin, Image as ImageIcon } from 'lucide-react';
import type { GisCoordinates, GisBoundingBox, GisGeoJsonFeature, SensorMetadata } from '../types';
import { GisMapViewer } from './GisMapViewer';

interface OpticalSarViewerProps {
  opticalUrl: string;
  sarUrl: string;
  opticalTitle?: string;
  sarTitle?: string;
  coordinates?: GisCoordinates;
  boundingBox?: GisBoundingBox;
  geojson?: GisGeoJsonFeature | GisGeoJsonFeature[];
  sensorMetadata?: SensorMetadata;
}

export const OpticalSarViewer: React.FC<OpticalSarViewerProps> = ({
  opticalUrl,
  sarUrl,
  opticalTitle = 'Optical Imagery (Sentinel-2 MSI)',
  sarTitle = 'SAR Imagery (Sentinel-1 C-Band Synthetic Aperture Radar)',
  coordinates,
  boundingBox,
  geojson,
  sensorMetadata,
}) => {
  const [activeTab, setActiveTab] = useState<'sensors' | 'gis-map'>('sensors');
  const [highlightTargets, setHighlightTargets] = useState(true);

  return (
    <div className="image-panel-card" style={{ marginBottom: 20 }}>
      <div className="image-panel-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <Layers size={18} color="var(--indigo)" />
          <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>Multi-Sensor Optical + SAR Inspector</span>

          {/* Tab Switcher: Multi-Sensor vs GIS Map */}
          <div style={{ display: 'flex', background: 'var(--bg-input)', borderRadius: 'var(--radius-full)', padding: 2, border: '1px solid var(--border-subtle)' }}>
            <button
              type="button"
              className={`btn btn-sm ${activeTab === 'sensors' ? 'btn-primary' : ''}`}
              style={{ padding: '3px 10px', fontSize: '0.74rem', borderRadius: 'var(--radius-full)' }}
              onClick={() => setActiveTab('sensors')}
            >
              <ImageIcon size={13} /> Dual Sensor
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
        </div>

        {activeTab === 'sensors' && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <button
              type="button"
              className={`btn btn-sm ${highlightTargets ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setHighlightTargets(!highlightTargets)}
            >
              <Radio size={13} />
              {highlightTargets ? 'Hide Radar Target Markers' : 'Highlight Radar Targets'}
            </button>
          </div>
        )}
      </div>

      {activeTab === 'gis-map' ? (
        <GisMapViewer
          coordinates={coordinates || { lat: 18.920, lng: 72.780, zoom: 11, locationName: 'Arabian Sea Maritime Channel & Approaches' }}
          boundingBox={boundingBox || { north: 19.12, south: 18.72, east: 72.98, west: 72.58 }}
          geojson={geojson}
          sensorMetadata={sensorMetadata}
          title="All-Weather Radar Target Location & Maritime Vector Footprint"
        />
      ) : (
        <div className="dual-viewer-grid" style={{ padding: 12 }}>
          {/* Optical Sensor Panel */}
          <div style={{ background: '#020617', borderRadius: 'var(--radius-md)', overflow: 'hidden', border: '1px solid var(--border-subtle)', position: 'relative' }}>
            <div style={{ padding: '8px 14px', background: 'var(--bg-card-elevated)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)' }}>
              <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--cyan)', display: 'flex', alignItems: 'center', gap: 6 }}>
                <Sun size={14} /> {opticalTitle}
              </span>
              <span style={{ fontSize: '0.72rem', color: 'var(--amber)', background: 'rgba(245, 158, 11, 0.12)', padding: '2px 8px', borderRadius: 4 }}>
                Cloud Obscuration: 76.4%
              </span>
            </div>

            <div style={{ height: 400, position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
              <img src={opticalUrl} alt="Optical Satellite Scene" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
            </div>

            <div style={{ padding: '10px 14px', background: 'var(--bg-input)', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              <strong>Optical Spectrum:</strong> Wavelength 400–700nm. Passive sensor measuring reflected solar illumination. Subject to severe cloud & atmospheric blockage.
            </div>
          </div>

          {/* SAR Sensor Panel */}
          <div style={{ background: '#020617', borderRadius: 'var(--radius-md)', overflow: 'hidden', border: '1px solid var(--border-subtle)', position: 'relative' }}>
            <div style={{ padding: '8px 14px', background: 'var(--bg-card-elevated)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)' }}>
              <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--indigo)', display: 'flex', alignItems: 'center', gap: 6 }}>
                <Radio size={14} /> {sarTitle}
              </span>
              <span style={{ fontSize: '0.72rem', color: 'var(--emerald)', background: 'rgba(16, 185, 129, 0.12)', padding: '2px 8px', borderRadius: 4 }}>
                All-Weather Penetration: 100%
              </span>
            </div>

            <div style={{ height: 400, position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
              <img src={sarUrl} alt="SAR Satellite Scene" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />

              {/* Visual radar targets highlight overlays */}
              {highlightTargets && (
                <div style={{ position: 'absolute', top: 12, right: 12, background: 'rgba(10, 15, 29, 0.85)', padding: '6px 10px', borderRadius: 6, border: '1px solid var(--cyan)', fontSize: '0.75rem', color: 'var(--cyan)', backdropFilter: 'blur(4px)' }}>
                  Radar Return: 4 Maritime Targets Detected
                </div>
              )}
            </div>

            <div style={{ padding: '10px 14px', background: 'var(--bg-input)', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              <strong>Microwave SAR:</strong> C-band 5.405 GHz active radar. Penetrates dense clouds and rain. Detects surface roughness and metallic dihedral returns.
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
