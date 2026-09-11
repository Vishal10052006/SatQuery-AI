import React, { useEffect, useMemo, useState } from 'react';
import {
  Activity,
  CheckCircle2,
  ChevronDown,
  CircleDot,
  Database,
  FileImage,
  Globe2,
  Layers3,
  Loader2,
  MapPin,
  MessageSquare,
  Network,
  Radar,
  Search,
  ShieldCheck,
  Sparkles,
  Upload,
  Workflow,
  XCircle,
} from 'lucide-react';
import type { AnalysisResponse, PipelineStage } from './types';
import { SAMPLE_SCENARIOS } from './services/mockData';
import { checkBackendHealth, executeSatelliteAnalysis } from './services/api';

interface SlotState { file?: File; previewUrl?: string; name: string; }

const EXAMPLE_QUERIES = [
  'What objects are visible in this image?',
  'What changed between these two images?',
  'Identify newly constructed areas.',
  'Compare optical and SAR information.',
];

const PIPELINE_STEPS = [
  { key: 'routing', label: 'Query Understanding', icon: Search },
  { key: 'routing', label: 'Intent Detection', icon: Workflow },
  { key: 'routing', label: 'Specialist Selection', icon: Network },
  { key: 'analyzing', label: 'Satellite Analysis', icon: Radar },
  { key: 'evidence', label: 'Evidence Generation', icon: Layers3 },
  { key: 'completed', label: 'Final Answer', icon: Sparkles },
] as const;

const SPECIALISTS = [
  { id: 'M1', name: 'EarthDial VQA', role: 'Visual question answering' },
  { id: 'M2', name: 'Change Detection', role: 'Bi-temporal analysis' },
  { id: 'M3', name: 'Optical + SAR', role: 'Multi-sensor fusion' },
  { id: 'M4', name: 'AI Agent', role: 'Intent + routing' },
  { id: 'M5', name: 'GIS / Evidence', role: 'Geospatial provenance' },
  { id: 'M6', name: 'Interface', role: 'Judge demo surface' },
] as const;

const stageRank: Record<PipelineStage, number> = { idle: 0, routing: 1, analyzing: 2, evidence: 3, completed: 4, error: -1 };
const toFilePreview = (file: File): string => URL.createObjectURL(file);
const confidenceLabel = (value: number): string => `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`;

const getSpecialist = (result: AnalysisResponse | null, query: string) => {
  if (result?.mode === 'change-detection' || /change|between|before|after|constructed/i.test(query)) {
    return { id: 'M2', name: 'Change Detection', reason: 'Temporal comparison requested' };
  }
  if (result?.mode === 'optical-sar' || /sar|optical|radar/i.test(query)) {
    return { id: 'M3', name: 'Optical + SAR', reason: 'Multi-sensor comparison requested' };
  }
  return { id: 'M1', name: 'EarthDial VQA', reason: 'Single-scene visual question requested' };
};

const stageLabel = (stage: PipelineStage) => ({
  idle: 'Ready', routing: 'Routing', analyzing: 'Analyzing', evidence: 'Evidence', completed: 'Complete', error: 'Error',
}[stage]);

export const App: React.FC = () => {
  const [isDemoMode, setIsDemoMode] = useState(true);
  const [backendOnline, setBackendOnline] = useState(false);
  const [query, setQuery] = useState('What changed in this area between the two satellite images?');
  const [slot1, setSlot1] = useState<SlotState>({ name: '' });
  const [slot2, setSlot2] = useState<SlotState>({ name: '' });
  const [pipelineStage, setPipelineStage] = useState<PipelineStage>('idle');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [currentResult, setCurrentResult] = useState<AnalysisResponse | null>(null);
  const [history, setHistory] = useState<AnalysisResponse[]>([]);
  const [activeEvidenceTab, setActiveEvidenceTab] = useState<'imagery' | 'map' | 'provenance'>('imagery');

  useEffect(() => {
    let alive = true;
    const healthCheck = async () => {
      const status = await checkBackendHealth();
      if (alive) setBackendOnline(status.isOnline);
    };
    void healthCheck();
    const interval = window.setInterval(() => void healthCheck(), 15000);
    return () => { alive = false; window.clearInterval(interval); };
  }, []);

  const specialist = useMemo(() => getSpecialist(currentResult, query), [currentResult, query]);
  const isRunning = !['idle', 'completed', 'error'].includes(pipelineStage);
  const needsPair = specialist.id === 'M2' || specialist.id === 'M3';
  const canAnalyze = Boolean(query.trim()) && Boolean(slot1.previewUrl) && (!needsPair || Boolean(slot2.previewUrl));

  const loadScenario = (scenarioId: string) => {
    const scenario = SAMPLE_SCENARIOS.find((item) => item.id === scenarioId);
    if (!scenario) return;
    setQuery(scenario.query);
    setCurrentResult(scenario.mockResponse);
    setErrorMessage(null);
    setPipelineStage('completed');
    if (scenario.images.primary) {
      setSlot1({ name: 'Sample satellite scene', previewUrl: scenario.images.primary });
      setSlot2({ name: '' });
    } else if (scenario.images.before && scenario.images.after) {
      setSlot1({ name: 'Before image', previewUrl: scenario.images.before });
      setSlot2({ name: 'After image', previewUrl: scenario.images.after });
    } else if (scenario.images.optical && scenario.images.sar) {
      setSlot1({ name: 'Optical image', previewUrl: scenario.images.optical });
      setSlot2({ name: 'SAR image', previewUrl: scenario.images.sar });
    }
  };

  const handleUpload = (slot: 1 | 2, file?: File) => {
    if (!file) return;
    const next = { file, name: file.name, previewUrl: toFilePreview(file) };
    if (slot === 1) setSlot1(next); else setSlot2(next);
    setCurrentResult(null); setPipelineStage('idle'); setErrorMessage(null);
  };

  const handleAnalyze = async () => {
    setErrorMessage(null);
    if (!canAnalyze) {
      setErrorMessage(needsPair ? 'Add both satellite inputs before running this specialist.' : 'Add a satellite image before running analysis.');
      return;
    }
    try {
      setPipelineStage('routing');
      await new Promise((resolve) => window.setTimeout(resolve, 260));
      setPipelineStage('analyzing');
      await new Promise((resolve) => window.setTimeout(resolve, 260));
      setPipelineStage('evidence');
      const mode = specialist.id === 'M2' ? 'change-detection' : specialist.id === 'M3' ? 'optical-sar' : 'image-understanding';
      const result = await executeSatelliteAnalysis({ mode, query: query.trim(), primaryImage: slot1, beforeImage: slot1, afterImage: slot2, opticalImage: slot1, sarImage: slot2 }, isDemoMode);
      setCurrentResult(result);
      setHistory((prev) => [result, ...prev].slice(0, 12));
      setPipelineStage('completed');
      setActiveEvidenceTab('imagery');
    } catch (error) {
      setPipelineStage('error');
      setErrorMessage(error instanceof Error ? error.message : 'Analysis pipeline failed.');
    }
  };

  const reset = () => {
    setQuery(''); setSlot1({ name: '' }); setSlot2({ name: '' }); setCurrentResult(null); setPipelineStage('idle'); setErrorMessage(null);
  };

  const confidence = currentResult?.confidence ?? 0;
  const mapCoordinates = currentResult?.coordinates;
  const hasMap = Boolean(mapCoordinates || currentResult?.boundingBox || currentResult?.geojson);

  return (
    <div className="judge-shell">
      <header className="judge-header">
        <div className="brand-lockup">
          <div className="brand-mark"><Radar size={22} /></div>
          <div><div className="brand-title">SATQUERY AI</div><div className="brand-subtitle">AI-Powered Natural Language Satellite Intelligence</div></div>
        </div>
        <div className="header-actions">
          <div className={`live-pill ${backendOnline ? 'live' : isDemoMode ? 'demo' : 'offline'}`}><CircleDot size={13} /><span>{backendOnline ? 'Live System' : isDemoMode ? 'Demo System' : 'Backend Offline'}</span></div>
          <button className="ghost-btn" type="button" onClick={() => setIsDemoMode((value) => !value)}>{isDemoMode ? 'Demo Mode' : 'Live Backend'}<ChevronDown size={15} /></button>
          <button className="ghost-btn" type="button" onClick={reset}><XCircle size={15} /> Reset</button>
        </div>
      </header>

      <main className="judge-content">
        <section className="hero-panel">
          <div><span className="eyebrow"><Sparkles size={14} /> JUDGE DEMO</span><h1>Ask a satellite question. Get an evidence-backed answer.</h1><p>Ask questions about satellite imagery in natural language and receive AI-driven analysis with visual and geospatial evidence.</p></div>
          <div className="hero-meta"><div><ShieldCheck size={15} /> Provenance-first output</div><div><Activity size={15} /> Agentic routing visible</div><div><Globe2 size={15} /> Geospatial context</div></div>
        </section>

        <section className="section-card upload-card">
          <div className="section-heading"><div><div className="section-title"><Upload size={18} /> Satellite Data</div><div className="section-caption">JPG / PNG / GeoTIFF · drag, drop, or load a judge-ready scenario</div></div><button className="scenario-btn" type="button" onClick={() => loadScenario('change-detection-flood')}>Load Demo Pair</button></div>
          <div className="upload-grid"><UploadTile label={needsPair ? 'Before Image' : 'Satellite Imagery'} slot={slot1} onFile={(file) => handleUpload(1, file)} />{needsPair && <UploadTile label={specialist.id === 'M3' ? 'SAR Image' : 'After Image'} slot={slot2} onFile={(file) => handleUpload(2, file)} />}</div>
        </section>

        <section className="section-card query-card">
          <div className="section-heading"><div><div className="section-title"><MessageSquare size={18} /> Ask SatQuery AI</div><div className="section-caption">The query is the primary control surface for the agentic system.</div></div></div>
          <textarea value={query} onChange={(event) => setQuery(event.target.value)} placeholder="What changed between these two satellite images?" aria-label="Satellite intelligence query" />
          <div className="example-row"><span>Try:</span>{EXAMPLE_QUERIES.map((example) => <button type="button" key={example} onClick={() => setQuery(example)}>{example}</button>)}</div>
          <div className="query-actions"><div className="query-status"><span className={`status-dot ${isRunning ? 'busy' : 'ready'}`} />{isRunning ? `${stageLabel(pipelineStage)} pipeline…` : `M4 routing ready → ${specialist.id}`}</div><button className="analyze-btn" type="button" onClick={handleAnalyze} disabled={isRunning || !canAnalyze}>{isRunning ? <Loader2 size={17} className="spin" /> : <Sparkles size={17} />}{isRunning ? 'Analyzing…' : 'Analyze'}</button></div>
          {errorMessage && <div className="error-banner"><XCircle size={15} /> {errorMessage}</div>}
        </section>

        <section className="section-card pipeline-card">
          <div className="section-heading compact"><div><div className="section-title"><Workflow size={18} /> AI Analysis Pipeline</div><div className="section-caption">Execution state exposed for judge visibility.</div></div><span className={`stage-chip ${pipelineStage}`}>{stageLabel(pipelineStage)}</span></div>
          <div className="pipeline-track">{PIPELINE_STEPS.map((step, index) => { const done = pipelineStage === 'completed' || stageRank[pipelineStage] > index; const active = (pipelineStage === 'routing' && index < 3) || (pipelineStage === 'analyzing' && index === 3) || (pipelineStage === 'evidence' && index === 4); const Icon = step.icon; return <React.Fragment key={`${step.key}-${step.label}`}><div className={`pipeline-node ${done ? 'done' : ''} ${active ? 'active' : ''}`}><div className="pipeline-icon">{done ? <CheckCircle2 size={16} /> : <Icon size={16} />}</div><span>{step.label}</span></div>{index < PIPELINE_STEPS.length - 1 && <div className={`pipeline-connector ${done ? 'done' : ''}`} />}</React.Fragment>; })}</div>
          <div className="specialist-row"><div><span className="mini-label">Selected Specialist</span><strong><Radar size={15} /> {specialist.id} — {specialist.name}</strong><span>{specialist.reason}</span></div><div className="routing-badge"><Network size={15} /> M4 Agent Controller</div></div>
        </section>

        {currentResult && <>
          <section className="result-grid">
            <div className="finding-card"><div className="finding-header"><div className="section-title"><Sparkles size={18} /> AI Finding</div><span className="success-tag"><CheckCircle2 size={13} /> SUCCESS</span></div><p className="finding-answer">{currentResult.answer}</p><div className="finding-footer"><div><span className="mini-label">Operational confidence</span><div className="confidence-meter"><span style={{ width: `${confidence * 100}%` }} /></div></div><strong>{confidenceLabel(confidence)}</strong></div>{currentResult.modelUsed && <div className="model-line"><Database size={14} /> {currentResult.modelUsed}</div>}</div>
            <div className="geo-card"><div className="finding-header"><div className="section-title"><MapPin size={18} /> Geospatial Evidence</div>{hasMap && <span className="success-tag"><Globe2 size={13} /> GEO-REFERENCED</span>}</div><div className="geo-stage"><div className="geo-gridlines" /><div className="geo-crosshair"><div className="crosshair-ring" /><MapPin size={28} /></div><div className="geo-badge">CHANGE REGION</div>{!hasMap && <div className="geo-empty">Coordinates or GeoJSON will appear here when returned by the analysis.</div>}</div><div className="geo-meta"><span>Location</span><strong>{mapCoordinates?.locationName ?? 'Analysis footprint'}</strong><span>Coordinates</span><strong>{mapCoordinates ? `${mapCoordinates.lat.toFixed(5)}, ${mapCoordinates.lng.toFixed(5)}` : '—'}</strong></div></div>
          </section>

          <section className="section-card evidence-card">
            <div className="evidence-tabs">{[['imagery','Visual Evidence'],['map','Geospatial Evidence'],['provenance','Evidence / Provenance']].map(([id, label]) => <button key={id} type="button" className={activeEvidenceTab === id ? 'active' : ''} onClick={() => setActiveEvidenceTab(id as typeof activeEvidenceTab)}>{label}</button>)}</div>
            {activeEvidenceTab === 'imagery' && <div className="imagery-grid"><ImageEvidenceCard label="BEFORE" src={currentResult.beforeImageUrl || currentResult.primaryImageUrl || slot1.previewUrl} /><ImageEvidenceCard label="AFTER" src={currentResult.afterImageUrl || currentResult.opticalImageUrl || slot2.previewUrl} /><ImageEvidenceCard label="CHANGE OVERLAY" src={currentResult.overlayImageUrl} overlay /></div>}
            {activeEvidenceTab === 'map' && <div className="map-proof-panel"><div className="map-proof-grid" /><div className="map-proof-marker"><MapPin size={20} /><span>{mapCoordinates?.locationName ?? 'Awaiting geospatial result'}</span></div><div className="map-proof-copy"><span className="mini-label">GIS / M5 evidence</span><h3>{hasMap ? 'Geospatial context attached to the answer.' : 'No geospatial artifact returned yet.'}</h3><p>Coordinates, bounding boxes, or GeoJSON returned by the backend are surfaced here as judge-facing evidence.</p></div></div>}
            {activeEvidenceTab === 'provenance' && <div className="provenance-grid"><ProvenanceGroup title="Source imagery" items={[slot1.previewUrl ? 'Primary / before image loaded' : 'Primary image not provided', slot2.previewUrl ? 'Secondary / after image loaded' : 'Secondary image not provided']} /><ProvenanceGroup title="Processing" items={currentResult.reasoningSteps.slice(0, 6)} /><ProvenanceGroup title="Specialist" items={[`${specialist.id} — ${specialist.name}`, `M4 reason: ${specialist.reason}`, currentResult.status.toUpperCase()]} /></div>}
          </section>

          <section className="trace-grid"><div className="section-card trace-card"><div className="section-title"><Search size={18} /> Execution Trace</div><div className="trace-list">{['Query parsed',`Intent routed: ${specialist.id === 'M2' ? 'CHANGE_DETECTION' : specialist.id === 'M3' ? 'OPTICAL_SAR' : 'IMAGE_UNDERSTANDING'}`,`${specialist.id} selected by M4`,needsPair ? 'Image inputs validated / compared' : 'Satellite image validated','Evidence generated','Response synthesized'].map((item) => <div key={item}><CheckCircle2 size={15} /> {item}</div>)}</div></div><div className="section-card trace-card"><div className="section-title"><Activity size={18} /> Runtime Summary</div><div className="runtime-grid"><div><span>Mode</span><strong>{isDemoMode ? 'Demo' : backendOnline ? 'Live' : 'Offline'}</strong></div><div><span>Status</span><strong>{currentResult.status.toUpperCase()}</strong></div><div><span>Latency</span><strong>{currentResult.executionTimeMs ? `${currentResult.executionTimeMs} ms` : '—'}</strong></div><div><span>History</span><strong>{history.length} run{history.length === 1 ? '' : 's'}</strong></div></div></div></section>
        </>}

        <section className="section-card specialist-card"><div className="section-heading compact"><div><div className="section-title"><ShieldCheck size={18} /> Specialist Status Dashboard</div><div className="section-caption">Six modules, one judge-facing workflow.</div></div></div><div className="specialist-grid">{SPECIALISTS.map((module) => <div key={module.id} className={`specialist-tile ${module.id === specialist.id ? 'selected' : ''}`}><div className="specialist-status"><span /> {module.id}</div><strong>{module.name}</strong><span>{module.role}</span></div>)}</div></section>
      </main>
    </div>
  );
};

interface UploadTileProps { label: string; slot: SlotState; onFile: (file: File) => void; }
const UploadTile: React.FC<UploadTileProps> = ({ label, slot, onFile }) => {
  const inputId = `upload-${label.replace(/\s+/g, '-').toLowerCase()}`;
  return <div className={`upload-tile ${slot.previewUrl ? 'filled' : ''}`}>{slot.previewUrl ? <><img src={slot.previewUrl} alt={label} /><div className="upload-overlay"><span>{slot.name}</span><label htmlFor={inputId}>Replace image</label></div></> : <label htmlFor={inputId} className="upload-empty"><FileImage size={28} /><strong>{label}</strong><span>Drag & drop image here</span><em>or click to upload</em></label>}<input id={inputId} type="file" accept="image/png,image/jpeg,image/tiff,.tif,.tiff" onChange={(event) => onFile(event.target.files?.[0] as File)} /></div>;
};

interface ImageEvidenceCardProps { label: string; src?: string; overlay?: boolean; }
const ImageEvidenceCard: React.FC<ImageEvidenceCardProps> = ({ label, src, overlay }) => <div className="evidence-image"><div className="evidence-image-label">{label}</div>{src ? <img src={src} alt={label} className={overlay ? 'overlay-image' : ''} /> : <div className="evidence-placeholder"><FileImage size={24} /><span>Waiting for image evidence</span></div>}</div>;

interface ProvenanceGroupProps { title: string; items: string[]; }
const ProvenanceGroup: React.FC<ProvenanceGroupProps> = ({ title, items }) => <div className="provenance-group"><div className="mini-label">{title}</div>{items.map((item) => <div key={item}><CheckCircle2 size={15} /> {item}</div>)}</div>;

export default App;
