import React, { useState } from 'react';
import { 
  Eye, 
  SplitSquareVertical, 
  Layers, 
  ArrowRight, 
  Compass, 
  Workflow, 
  CheckCircle2,
  Sparkles,
  Zap,
  Terminal,
  Send,
  Loader2,
  Clock,
  History,
  FileText
} from 'lucide-react';
import type { ViewMode, PipelineStage, AnalysisResponse } from '../types';
import { SAMPLE_SCENARIOS } from '../services/mockData';

interface DashboardOverviewProps {
  onSelectMode: (mode: ViewMode) => void;
  onLaunchDemoScenario: (scenarioId: string) => void;
  currentResult: AnalysisResponse | null;
  pipelineStage: PipelineStage;
  history: AnalysisResponse[];
  onOpenReport?: () => void;
}

export const DashboardOverview: React.FC<DashboardOverviewProps> = ({
  onSelectMode,
  onLaunchDemoScenario,
  currentResult,
  pipelineStage,
  history,
  onOpenReport,
}) => {
  const [quickMode, setQuickMode] = useState<ViewMode>('image-understanding');
  const [quickQuery, setQuickQuery] = useState<string>(
    'Identify the major land-use features in this image.'
  );

  const landUseScenario = SAMPLE_SCENARIOS.find((s) => s.id === 'land-use-analysis');
  const floodScenario = SAMPLE_SCENARIOS.find((s) => s.id === 'change-detection-flood');
  const sarScenario = SAMPLE_SCENARIOS.find((s) => s.id === 'optical-sar-comparison');

  const handleModeTabChange = (mode: ViewMode) => {
    setQuickMode(mode);
    if (mode === 'image-understanding') {
      setQuickQuery('Identify the major land-use features in this image.');
    } else if (mode === 'change-detection') {
      setQuickQuery('What changed between these two images?');
    } else if (mode === 'optical-sar') {
      setQuickQuery('Compare the optical and SAR imagery.');
    }
  };

  const handleRunQuickAnalysis = () => {
    if (quickMode === 'image-understanding') {
      onLaunchDemoScenario('land-use-analysis');
    } else if (quickMode === 'change-detection') {
      onLaunchDemoScenario('change-detection-flood');
    } else if (quickMode === 'optical-sar') {
      onLaunchDemoScenario('optical-sar-comparison');
    }
  };

  const isLoading = pipelineStage !== 'idle' && pipelineStage !== 'completed' && pipelineStage !== 'error';

  return (
    <div>
      {/* High-Tech Command Center Hero Banner */}
      <section className="hero-banner">
        <div className="hero-tagline">
          <Sparkles size={13} />
          Smart India Hackathon 2026 • Problem Statement SIH26167
        </div>
        <h1 className="hero-heading">SatQuery AI: Earth Observation Intelligence Engine</h1>
        <p className="hero-description">
          An interactive, mission-grade geospatial intelligence dashboard bridging natural language queries 
          with multi-spectral optical and Synthetic Aperture Radar (SAR) satellite data. Inspect semantic land-use, 
          track bi-temporal environmental disasters with curtain-swipe sliders, and penetrate cloud obscuration.
        </p>

        {/* KPI Metric Counter Tiles */}
        <div className="hero-stats-row">
          <div className="hero-stat-card">
            <span className="hero-stat-label">Analysis Pipelines</span>
            <span className="hero-stat-val" style={{ color: 'var(--cyan-bright)' }}>03 Modes</span>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 2 }}>VLM • Change • SAR Fusion</span>
          </div>

          <div className="hero-stat-card">
            <span className="hero-stat-label">Ground Sampling Distance</span>
            <span className="hero-stat-val" style={{ color: 'var(--emerald-bright)' }}>10m / px</span>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 2 }}>Sentinel-2 L2A Multispectral</span>
          </div>

          <div className="hero-stat-card">
            <span className="hero-stat-label">Cloud Penetration</span>
            <span className="hero-stat-val" style={{ color: 'var(--indigo)' }}>100% SAR</span>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 2 }}>C-Band 5.4 GHz Active Radar</span>
          </div>

          <div className="hero-stat-card">
            <span className="hero-stat-label">End-to-End Latency</span>
            <span className="hero-stat-val" style={{ color: '#f59e0b' }}>&lt; 1.8s</span>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 2 }}>Async FastAPI + GPU Inference</span>
          </div>
        </div>
      </section>

      {/* Interactive Dashboard Quick Analysis Station */}
      <section className="card" style={{ marginBottom: 28, borderColor: 'var(--border-highlight)' }}>
        <div className="card-header">
          <div>
            <div className="card-title">
              <Terminal size={18} color="var(--cyan)" />
              Command Center Quick Launch Station
            </div>
            <div className="card-subtitle">
              Select an analysis mode, input your natural language query, and run satellite inference directly.
            </div>
          </div>

          <div style={{ display: 'flex', gap: 6, background: 'var(--bg-input)', padding: 3, borderRadius: 'var(--radius-full)', border: '1px solid var(--border-medium)' }}>
            <button
              type="button"
              className={`btn btn-sm ${quickMode === 'image-understanding' ? 'btn-primary' : ''}`}
              style={{ borderRadius: 'var(--radius-full)', fontSize: '0.76rem', padding: '4px 12px' }}
              onClick={() => handleModeTabChange('image-understanding')}
            >
              <Eye size={13} /> Single VLM
            </button>
            <button
              type="button"
              className={`btn btn-sm ${quickMode === 'change-detection' ? 'btn-primary' : ''}`}
              style={{ borderRadius: 'var(--radius-full)', fontSize: '0.76rem', padding: '4px 12px' }}
              onClick={() => handleModeTabChange('change-detection')}
            >
              <SplitSquareVertical size={13} /> Change Detection
            </button>
            <button
              type="button"
              className={`btn btn-sm ${quickMode === 'optical-sar' ? 'btn-primary' : ''}`}
              style={{ borderRadius: 'var(--radius-full)', fontSize: '0.76rem', padding: '4px 12px' }}
              onClick={() => handleModeTabChange('optical-sar')}
            >
              <Layers size={13} /> Optical + SAR
            </button>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16, alignItems: 'center' }}>
          <div>
            <label style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', display: 'block', marginBottom: 6, fontWeight: 600 }}>
              Natural Language Satellite Inquiry:
            </label>
            <input
              type="text"
              value={quickQuery}
              onChange={(e) => setQuickQuery(e.target.value)}
              placeholder="e.g., Identify land-use features or flood damage..."
              style={{
                width: '100%',
                background: 'var(--bg-input)',
                border: '1px solid var(--border-medium)',
                borderRadius: 'var(--radius-sm)',
                padding: '10px 14px',
                color: '#fff',
                fontSize: '0.86rem',
                outline: 'none',
              }}
            />
          </div>

          <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end' }}>
            <button
              type="button"
              className="btn btn-primary"
              onClick={handleRunQuickAnalysis}
              disabled={isLoading}
              style={{ flex: 1, height: 42 }}
            >
              {isLoading ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  Running SatQuery AI...
                </>
              ) : (
                <>
                  <Send size={15} />
                  Analyze & Launch Viewer
                </>
              )}
            </button>

            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => onSelectMode(quickMode)}
              style={{ height: 42 }}
              title="Open full interactive workspace with custom upload"
            >
              Open Full Studio
            </button>
          </div>
        </div>

        {/* Live Result Preview Card (if an analysis has been executed) */}
        {currentResult && (
          <div
            style={{
              marginTop: 18,
              paddingTop: 16,
              borderTop: '1px solid var(--border-subtle)',
              background: 'var(--bg-input)',
              padding: '14px 18px',
              borderRadius: 'var(--radius-md)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span className="badge badge-emerald">Latest Intelligence Output</span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  Confidence: {currentResult.confidence ? Math.round(currentResult.confidence * 100) : 92}%
                </span>
              </div>

              <div style={{ display: 'flex', gap: 8 }}>
                {onOpenReport && (
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={onOpenReport}
                    style={{ fontSize: '0.74rem' }}
                  >
                    <FileText size={12} />
                    Report
                  </button>
                )}
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={() => onSelectMode(currentResult.mode)}
                  style={{ background: 'var(--cyan)', color: '#030e17', fontWeight: 600, fontSize: '0.74rem' }}
                >
                  Inspect in {currentResult.mode.replace('-', ' ')} ➔
                </button>
              </div>
            </div>

            <div style={{ fontSize: '0.86rem', color: '#f1f5f9', lineHeight: 1.5, marginBottom: 4 }}>
              <strong>Summary: </strong>{currentResult.resultSummary}
            </div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {currentResult.answer}
            </div>
          </div>
        )}
      </section>

      {/* Interactive Scenario Cards Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 18 }}>
        <div>
          <h2 style={{ fontSize: '1.2rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: 10 }}>
            <Compass size={22} color="var(--cyan)" />
            Core Operational Workflows & SIH Scenarios
          </h2>
          <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)' }}>
            Select any scenario below to automatically load the satellite assets and trigger the analysis pipeline.
          </p>
        </div>

        <div className="badge badge-cyan" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <Zap size={13} />
          Interactive Demo Ready
        </div>
      </div>

      {/* Visual Scenario Cards with Satellite Imagery Thumbnails */}
      <div className="dashboard-grid">
        {/* Scenario 1: Land Use Understanding */}
        <div 
          className="scenario-card"
          onClick={() => onLaunchDemoScenario('land-use-analysis')}
        >
          <div className="scenario-thumb-wrapper">
            <img 
              src={landUseScenario?.images.primary} 
              alt="Land-Use Scene Preview" 
              className="scenario-thumb-img" 
            />
            <div className="scenario-thumb-overlay" />
            <div className="scenario-thumb-badge">
              <span className="badge badge-cyan">
                <Eye size={12} /> M1: VLM Reasoning
              </span>
            </div>
          </div>

          <div className="scenario-body">
            <div className="scenario-title">Single Image Understanding</div>
            <div className="scenario-desc">
              Extract semantic land-cover regimes, coastal estuarine boundaries, built infrastructure, and agricultural plots via Geospatial VLM.
            </div>

            <div className="scenario-prompt-box">
              <span style={{ color: 'var(--cyan)', fontWeight: 700 }}>Query: </span>
              "Identify the major land-use features in this image."
            </div>

            <div className="scenario-footer">
              <span style={{ color: 'var(--cyan-bright)' }}>Launch VLM Analysis</span>
              <ArrowRight size={16} color="var(--cyan-bright)" />
            </div>
          </div>
        </div>

        {/* Scenario 2: Change Detection */}
        <div 
          className="scenario-card"
          onClick={() => onLaunchDemoScenario('change-detection-flood')}
        >
          <div className="scenario-thumb-wrapper">
            <img 
              src={floodScenario?.images.after} 
              alt="Flood Scene Preview" 
              className="scenario-thumb-img" 
            />
            <div className="scenario-thumb-overlay" />
            <div className="scenario-thumb-badge">
              <span className="badge badge-emerald">
                <SplitSquareVertical size={12} /> M2: Siamese ChangeNet
              </span>
            </div>
          </div>

          <div className="scenario-body">
            <div className="scenario-title">Bi-Temporal Change Detection</div>
            <div className="scenario-desc">
              Compare pre- and post-event satellite imagery with interactive split curtain swipe sliders and highlighted inundation masks.
            </div>

            <div className="scenario-prompt-box">
              <span style={{ color: 'var(--emerald)', fontWeight: 700 }}>Query: </span>
              "What changed between these two images?"
            </div>

            <div className="scenario-footer">
              <span style={{ color: 'var(--emerald-bright)' }}>Launch Change Detection</span>
              <ArrowRight size={16} color="var(--emerald-bright)" />
            </div>
          </div>
        </div>

        {/* Scenario 3: Optical + SAR */}
        <div 
          className="scenario-card"
          onClick={() => onLaunchDemoScenario('optical-sar-comparison')}
        >
          <div className="scenario-thumb-wrapper">
            <img 
              src={sarScenario?.images.sar} 
              alt="SAR Radar Scene Preview" 
              className="scenario-thumb-img" 
            />
            <div className="scenario-thumb-overlay" />
            <div className="scenario-thumb-badge">
              <span className="badge badge-cyan" style={{ borderColor: 'var(--indigo)', background: 'rgba(99, 102, 241, 0.2)', color: '#a5b4fc' }}>
                <Layers size={12} /> M3: Optical + SAR Fusion
              </span>
            </div>
          </div>

          <div className="scenario-body">
            <div className="scenario-title">Optical + SAR Multi-Sensor Fusion</div>
            <div className="scenario-desc">
              Overcome 76%+ dense cloud obstruction by pairing optical imagery with synthetic aperture radar (SAR) microwave backscatter.
            </div>

            <div className="scenario-prompt-box">
              <span style={{ color: 'var(--indigo)', fontWeight: 700 }}>Query: </span>
              "Compare the optical and SAR imagery."
            </div>

            <div className="scenario-footer">
              <span style={{ color: '#a5b4fc' }}>Launch Multi-Sensor Fusion</span>
              <ArrowRight size={16} color="#a5b4fc" />
            </div>
          </div>
        </div>
      </div>

      {/* Recent History Strip (if any) */}
      {history.length > 0 && (
        <div style={{ marginTop: 24, marginBottom: 24 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.95rem', fontWeight: 600 }}>
              <History size={16} color="var(--cyan)" />
              Recent Mission Runs
            </div>
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={() => onSelectMode('history')}
              style={{ fontSize: '0.75rem' }}
            >
              View Full History ({history.length})
            </button>
          </div>

          <div style={{ display: 'flex', gap: 12, overflowX: 'auto', paddingBottom: 6 }}>
            {history.slice(0, 3).map((item) => (
              <div
                key={item.id}
                onClick={() => onSelectMode('history')}
                style={{
                  background: 'var(--bg-card-elevated)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 'var(--radius-md)',
                  padding: '10px 14px',
                  minWidth: 260,
                  cursor: 'pointer',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4, fontSize: '0.72rem' }}>
                  <span className="badge badge-cyan" style={{ textTransform: 'capitalize' }}>
                    {item.mode.replace('-', ' ')}
                  </span>
                  <span style={{ color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 4 }}>
                    <Clock size={11} /> {new Date(item.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-primary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  "{item.query}"
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Team Pipeline Architecture Visualization */}
      <div className="pipeline-box">
        <div className="pipeline-header">
          <div>
            <div style={{ fontSize: '1.05rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: 10 }}>
              <Workflow size={20} color="var(--cyan)" />
              End-to-End System Pipeline Architecture
            </div>
            <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
              Decoupled modular pipeline coordinating the 6 specialized SIH team modules
            </div>
          </div>
          <div className="badge badge-emerald">
            <CheckCircle2 size={13} />
            Frontend Module (M6) Active
          </div>
        </div>

        <div className="pipeline-steps">
          <div className="pipeline-step">
            <div className="pipeline-step-badge">Module M4</div>
            <div className="pipeline-step-title">Agentic Routing</div>
            <div className="pipeline-step-desc">Classifies natural language intent and routes query to targeted vision pipelines.</div>
          </div>
          <div className="pipeline-arrow">➔</div>

          <div className="pipeline-step">
            <div className="pipeline-step-badge">Modules M1 / M2 / M3</div>
            <div className="pipeline-step-title">AI Vision Core</div>
            <div className="pipeline-step-desc">VLM reasoning, Siamese change segmentation, & SAR microwave fusion.</div>
          </div>
          <div className="pipeline-arrow">➔</div>

          <div className="pipeline-step">
            <div className="pipeline-step-badge">Module M5</div>
            <div className="pipeline-step-title">FastAPI Backend</div>
            <div className="pipeline-step-desc">High-throughput async server handling image inputs and API response contracts.</div>
          </div>
          <div className="pipeline-arrow">➔</div>

          <div className="pipeline-step active-layer">
            <div className="pipeline-step-badge" style={{ color: 'var(--cyan-bright)' }}>Module M6 (Here)</div>
            <div className="pipeline-step-title">GIS / UI Visualization</div>
            <div className="pipeline-step-desc">Interactive dashboard, swipe sliders, overlays, and confidence verification.</div>
          </div>
        </div>
      </div>
    </div>
  );
};
