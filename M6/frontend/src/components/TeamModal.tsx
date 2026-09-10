import { X, Cpu } from 'lucide-react';

interface TeamModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const TeamModal: React.FC<TeamModalProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  const modules = [
    {
      id: 'M1',
      title: 'VLM / Visual Question Answering (VQA)',
      scope: 'Multi-modal vision-language models for satellite object localization, zero-shot question answering, and spatial reasoning.',
      status: 'Pipeline Integrated',
    },
    {
      id: 'M2',
      title: 'Computer Vision / Change Detection',
      scope: 'Siamese neural networks, bi-temporal differential indexing (NDWI/NDVI), and flood/damage segmentation masks.',
      status: 'Pipeline Integrated',
    },
    {
      id: 'M3',
      title: 'Optical + SAR Sensor Fusion',
      scope: 'Cross-sensor calibration between Sentinel-2 optical imagery and Sentinel-1 C-Band synthetic aperture radar.',
      status: 'Pipeline Integrated',
    },
    {
      id: 'M4',
      title: 'Agentic AI / Intelligent Routing',
      scope: 'Natural language intent classification, orchestrating queries across domain-specific vision modules and synthesis.',
      status: 'Pipeline Integrated',
    },
    {
      id: 'M5',
      title: 'Backend / FastAPI / Data Pipeline',
      scope: 'High-throughput async FastAPI server, image preprocessing, GeoTIFF tiling, and model inference dispatch.',
      status: 'Configured (Live / Mock API Switcher)',
    },
    {
      id: 'M6',
      title: 'Frontend / GIS Dashboard (This Module)',
      scope: 'Interactive geospatial dashboard, before/after swipe curtain slider, optical-SAR inspector, confidence & evidence visualization.',
      status: 'Active & Demo-Ready',
      isCurrent: true,
    },
  ];

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
        backgroundColor: 'rgba(5, 8, 16, 0.85)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 100,
        backdropFilter: 'blur(6px)',
        padding: 20,
      }}
      onClick={onClose}
    >
      <div
        className="card"
        style={{
          maxWidth: 680,
          width: '100%',
          maxHeight: '90vh',
          overflowY: 'auto',
          background: 'var(--bg-card)',
          borderColor: 'var(--border-highlight)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="card-header" style={{ marginBottom: 12 }}>
          <div>
            <div className="card-title" style={{ fontSize: '1.15rem' }}>
              <Cpu size={20} color="var(--cyan)" />
              SatQuery AI • SIH Team Module Architecture
            </div>
            <div className="card-subtitle">
              Problem Statement SIH26167 • 6-Member Team Division of Responsibilities
            </div>
          </div>
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={onClose}
            style={{ padding: '4px 8px' }}
          >
            <X size={16} />
          </button>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginTop: 14 }}>
          {modules.map((m) => (
            <div
              key={m.id}
              style={{
                background: m.isCurrent ? 'rgba(6, 182, 212, 0.08)' : 'var(--bg-card-elevated)',
                border: `1px solid ${m.isCurrent ? 'var(--cyan)' : 'var(--border-subtle)'}`,
                borderRadius: 'var(--radius-md)',
                padding: '12px 16px',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 600, color: m.isCurrent ? 'var(--cyan)' : 'var(--text-primary)' }}>
                  [{m.id}] {m.title}
                </span>
                <span className={m.isCurrent ? 'badge badge-cyan' : 'badge badge-emerald'} style={{ fontSize: '0.68rem' }}>
                  {m.status}
                </span>
              </div>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                {m.scope}
              </div>
            </div>
          ))}
        </div>

        <div style={{ marginTop: 20, textAlign: 'right' }}>
          <button type="button" className="btn btn-primary btn-sm" onClick={onClose}>
            Close Architecture Overview
          </button>
        </div>
      </div>
    </div>
  );
};
