import React, { useEffect, useMemo, useState } from 'react';
import {
  Activity,
  ArrowUpRight,
  Check,
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
  X,
} from 'lucide-react';
import type { AnalysisResponse, PipelineStage } from './types';
import { SAMPLE_SCENARIOS } from './services/mockData';
import { API_BASE_URL, checkBackendHealth, executeSatelliteAnalysis } from './services/api';

interface SlotState {
  file?: File;
  previewUrl?: string;
  name: string;
}

const EXAMPLE_QUERIES = [
  'What changed between these two images, and where are the changed regions?',
  'What objects are visible in this image?',
  'Identify newly constructed areas.',
  'Compare optical and SAR information.',
];

const PIPELINE_STEPS = [
  { label: 'Query', icon: Search },
  { label: 'Intent', icon: Workflow },
  { label: 'Route', icon: Network },
  { label: 'Specialist', icon: Radar },
  { label: 'Evidence', icon: Layers3 },
  { label: 'Answer', icon: Sparkles },
] as const;

const SPECIALISTS = [
  { id: 'M1', name: 'EarthDial VQA', role: 'Visual understanding' },
  { id: 'M2', name: 'Change Detection', role: 'Bi-temporal analysis' },
  { id: 'M3', name: 'Optical + SAR', role: 'Sensor fusion' },
  { id: 'M4', name: 'AI Agent', role: 'Intent + routing' },
  { id: 'M5', name: 'GIS / Evidence', role: 'Spatial provenance' },
  { id: 'M6', name: 'Interface', role: 'Judge experience' },
] as const;

const stageRank: Record<PipelineStage, number> = { idle: 0, routing: 1, analyzing: 2, evidence: 3, completed: 4, error: -1 };
const stageLabel = (stage: PipelineStage): string => ({ idle: 'Ready', routing: 'Routing', analyzing: 'Analyzing', evidence: 'Building evidence', completed: 'Complete', error: 'Error' }[stage]);

const specialistFor = (result: AnalysisResponse | null, query: string) => {
  const tool = result?.toolResults?.find((item) => /m1_vqa|m2_change_detection|m3_optical_sar/.test(item.tool))?.tool;
  if (tool?.includes('m2_change_detection') || /change|between|before|after|constructed|construction/i.test(query)) return { id: 'M2', name: 'Change Detection', reason: 'Temporal comparison requested' };
  if (tool?.includes('m3_optical_sar') || /sar|optical|radar/i.test(query)) return { id: 'M3', name: 'Optical + SAR', reason: 'Multi-sensor comparison requested' };
  return { id: 'M1', name: 'EarthDial VQA', reason: 'Single-scene visual question requested' };
};

const confidenceLabel = (value: number): string => `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`;
const toFilePreview = (file: File): string => URL.createObjectURL(file);

const displayValue = (value: unknown): string => {
  if (Array.isArray(value)) return `${value.length} item${value.length === 1 ? '' : 's'}`;
  if (typeof value === 'object' && value !== null) return 'Structured output';
  return String(value);
};

const normalizeStatus = (value: unknown): string => String(value ?? '').trim().toLowerCase();
const statusLabel = (result: AnalysisResponse | null): string => {
  const status = normalizeStatus(result?.status);
  if (status === 'completed' || status === 'success') return 'SUCCESS';
  if (status === 'partial' || status === 'degraded' || status === 'fallback_baseline') return 'DEGRADED';
  if (status === 'error' || status === 'failed') return 'FAILED';
  return 'PENDING';
};
const routeLabel = (result: AnalysisResponse | null, specialistId: string, demo: boolean): string => {
  if (demo) return 'Demo response';
  const tools = result?.toolResults ?? [];
  if (tools.length === 0) return 'M4 response';
  return `M4 → ${tools.map((item) => item.tool.replaceAll('_', ' ')).join(' → ')}`;
};

export const App: React.FC = () => {
  const [isDemoMode, setIsDemoMode] = useState(false);
  const [backendOnline, setBackendOnline] = useState(false);
  const [query, setQuery] = useState(EXAMPLE_QUERIES[0]);
  const [slot1, setSlot1] = useState<SlotState>({ name: '' });
  const [slot2, setSlot2] = useState<SlotState>({ name: '' });
  const [pipelineStage, setPipelineStage] = useState<PipelineStage>('idle');
  const [currentResult, setCurrentResult] = useState<AnalysisResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [history, setHistory] = useState<AnalysisResponse[]>([]);

  useEffect(() => {
    let active = true;
    const checkHealth = async () => {
      const result = await checkBackendHealth();
      if (active) setBackendOnline(result.isOnline);
    };
    void checkHealth();
    const timer = window.setInterval(() => void checkHealth(), 15000);
    return () => { active = false; window.clearInterval(timer); };
  }, []);

  const specialist = useMemo(() => specialistFor(currentResult, query), [currentResult, query]);
  const isRunning = ['routing', 'analyzing', 'evidence'].includes(pipelineStage);
  const needsPair = specialist.id === 'M2' || specialist.id === 'M3';
  const canAnalyze = Boolean(query.trim()) && Boolean(slot1.previewUrl) && (!needsPair || Boolean(slot2.previewUrl));
  const confidence = currentResult?.confidence ?? 0;
  const actualTool = currentResult?.toolResults?.[currentResult.toolResults.length - 1];
  const resultStatus = normalizeStatus(currentResult?.status);
  const badge = statusLabel(currentResult);
  const mapArtifact = currentResult?.artifactUrls?.find((url) => /\/map\.html(?:$|\?)/i.test(url));
  const geoJsonArtifact = currentResult?.artifactUrls?.find((url) => /\.geojson(?:$|\?)/i.test(url));
  const coordinates = currentResult?.coordinates;

  const externalMapUrl = useMemo(() => {
    if (!coordinates) return undefined;
    const { lat, lng } = coordinates;
    const bbox = currentResult?.boundingBox;
    const west = bbox?.west ?? lng - 0.12;
    const south = bbox?.south ?? lat - 0.08;
    const east = bbox?.east ?? lng + 0.12;
    const north = bbox?.north ?? lat + 0.08;
    if (![lat, lng, west, south, east, north].every(Number.isFinite)) return undefined;
    if (lat < -90 || lat > 90 || lng < -180 || lng > 180) return undefined;
    return `https://www.openstreetmap.org/export/embed.html?bbox=${west},${south},${east},${north}&layer=mapnik&marker=${lat},${lng}`;
  }, [coordinates, currentResult?.boundingBox]);

  const loadScenario = () => {
    const scenario = SAMPLE_SCENARIOS.find((item) => item.id === 'change-detection-flood');
    if (!scenario) return;
    setIsDemoMode(true);
    setQuery(scenario.query);
    setCurrentResult(scenario.mockResponse);
    setPipelineStage('completed');
    setErrorMessage(null);
    setSlot1({ name: 'Before image', previewUrl: scenario.images.before ?? '' });
    setSlot2({ name: 'After image', previewUrl: scenario.images.after ?? '' });
  };

  const handleUpload = (slot: 1 | 2, file?: File) => {
    if (!file) return;
    const next = { file, name: file.name, previewUrl: toFilePreview(file) };
    if (slot === 1) setSlot1(next); else setSlot2(next);
    setCurrentResult(null);
    setPipelineStage('idle');
    setErrorMessage(null);
  };

  const handleAnalyze = async () => {
    setErrorMessage(null);
    if (!canAnalyze) {
      setErrorMessage(needsPair ? 'Add both source images before analysis.' : 'Add a satellite image before analysis.');
      return;
    }
    try {
      setCurrentResult(null);
      setPipelineStage('routing');
      await new Promise((resolve) => window.setTimeout(resolve, 180));
      setPipelineStage('analyzing');
      const mode = specialist.id === 'M2' ? 'change-detection' : specialist.id === 'M3' ? 'optical-sar' : 'image-understanding';
      const result = await executeSatelliteAnalysis({ mode, query: query.trim(), primaryImage: slot1, beforeImage: slot1, afterImage: slot2, opticalImage: slot1, sarImage: slot2 }, isDemoMode);
      setPipelineStage('evidence');
      setCurrentResult(result);
      setHistory((items) => [result, ...items].slice(0, 8));
      setPipelineStage(result.status === 'error' ? 'error' : 'completed');
    } catch (error) {
      setPipelineStage('error');
      setErrorMessage(error instanceof Error ? error.message : 'Analysis pipeline failed.');
    }
  };

  const reset = () => {
    setQuery(EXAMPLE_QUERIES[0]);
    setSlot1({ name: '' });
    setSlot2({ name: '' });
    setCurrentResult(null);
    setPipelineStage('idle');
    setErrorMessage(null);
  };

  const scalarData = actualTool
    ? Object.entries(actualTool.data).filter(([key, value]) => ['string', 'number', 'boolean'].includes(typeof value) && !/path|before|after|reference|model|answer/i.test(key)).slice(0, 8)
    : [];

  const sourceImages = needsPair
    ? [{ label: specialist.id === 'M3' ? 'OPTICAL' : 'BEFORE', src: slot1.previewUrl }, { label: specialist.id === 'M3' ? 'SAR' : 'AFTER', src: slot2.previewUrl }]
    : [{ label: 'SATELLITE IMAGE', src: slot1.previewUrl }];

  const trace = currentResult?.reasoningSteps?.length
    ? currentResult.reasoningSteps
    : actualTool
      ? [`${actualTool.tool.replaceAll('_', ' ')} → ${actualTool.status}`]
      : [];

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand"><div className="brand-icon"><Radar size={18} /></div><div><strong>SATQUERY <span>AI</span></strong><small>Natural-language satellite intelligence</small></div></div>
        <div className="topbar-actions">
          <span className={`system-status ${backendOnline ? 'online' : 'offline'}`}><CircleDot size={11} /> {backendOnline ? 'LIVE SYSTEM' : 'BACKEND OFFLINE'}</span>
          <button className={`mode-button ${isDemoMode ? 'active' : ''}`} type="button" onClick={() => setIsDemoMode((value) => !value)}>{isDemoMode ? 'Demo data' : 'Live backend'} <ChevronDown size={13} /></button>
          <button className="icon-button" type="button" onClick={reset} aria-label="Reset"><X size={16} /></button>
        </div>
      </header>

      <main className="workspace">
        <section className="intro">
          <div><p className="kicker"><Sparkles size={13} /> JUDGE WORKSPACE</p><h1>Ask. Analyze. <em>See the evidence.</em></h1><p className="intro-copy">Ask questions about satellite imagery in natural language and receive AI-driven analysis with visual and geospatial evidence.</p></div>
          <div className="intro-proof"><span><ShieldCheck size={14} /> Real backend artifacts</span><span><Network size={14} /> M4 routing</span><span><Globe2 size={14} /> GIS-ready</span></div>
        </section>

        <section className="command-grid">
          <div className="panel data-panel">
            <div className="panel-head"><div><p className="overline">01 · INPUT</p><h2>Satellite data</h2></div><button className="text-button" type="button" onClick={loadScenario}>Load demo pair <ArrowUpRight size={14} /></button></div>
            <div className={`source-grid ${needsPair ? 'pair' : 'single'}`}><UploadTile label={needsPair ? (specialist.id === 'M3' ? 'Optical image' : 'Before image') : 'Satellite image'} slot={slot1} onFile={(file) => handleUpload(1, file)} />{needsPair && <UploadTile label={specialist.id === 'M3' ? 'SAR image' : 'After image'} slot={slot2} onFile={(file) => handleUpload(2, file)} />}</div>
          </div>

          <div className="panel query-panel">
            <div className="panel-head"><div><p className="overline">02 · QUERY</p><h2>Ask SatQuery</h2></div><span className="route-hint"><span /> M4 → {specialist.id}</span></div>
            <div className="query-box"><MessageSquare size={16} /><textarea value={query} onChange={(event) => setQuery(event.target.value)} aria-label="Satellite intelligence query" placeholder="What changed between these images?" /></div>
            <div className="suggestions">{EXAMPLE_QUERIES.slice(0, 3).map((example) => <button key={example} type="button" onClick={() => setQuery(example)}>{example}</button>)}</div>
            <div className="query-footer"><span className="input-note">{needsPair ? 'Two observations required' : 'One observation required'} · JPG / PNG / GeoTIFF</span><button className="primary-button" type="button" onClick={handleAnalyze} disabled={isRunning || !canAnalyze}>{isRunning ? <Loader2 className="spin" size={16} /> : <Sparkles size={16} />}{isRunning ? 'Analyzing' : 'Analyze'}</button></div>
            {errorMessage && <div className="error-line">{errorMessage}</div>}
          </div>
        </section>

        <section className="pipeline-strip">
          <div className="pipeline-title"><span className="overline">03 · AGENT</span><strong>Execution</strong></div>
          <div className="pipeline-steps">{PIPELINE_STEPS.map((step, index) => { const done = pipelineStage === 'completed' || stageRank[pipelineStage] > index; const active = pipelineStage !== 'completed' && ((pipelineStage === 'routing' && index < 3) || (pipelineStage === 'analyzing' && index === 3) || (pipelineStage === 'evidence' && index === 4)); const Icon = step.icon; return <div className={`pipeline-step ${done ? 'done' : ''} ${active ? 'active' : ''}`} key={step.label}><span>{done ? <Check size={13} /> : <Icon size={13} />}</span><small>{step.label}</small>{index < PIPELINE_STEPS.length - 1 && <i />}</div>; })}</div>
          <span className={`pipeline-state ${pipelineStage}`}>{stageLabel(pipelineStage)}</span>
        </section>

        {currentResult && <>
          <section className="result-header"><div><p className="overline">04 · RESULT</p><h2>Analysis result</h2></div><div className="result-meta"><span><Database size={13} /> {routeLabel(currentResult, specialist.id, isDemoMode)}</span>{currentResult.executionTimeMs != null && <span>{currentResult.executionTimeMs} ms</span>}</div></section>

          <section className="result-grid">
            <article className="answer-card"><div className="card-label"><Sparkles size={14} /> AI finding <span className={`result-badge ${badge.toLowerCase()}`}>{badge}</span></div><p className="answer">{currentResult.answer || (resultStatus === 'error' ? currentResult.error || 'Analysis failed.' : 'Analysis completed.')}</p><div className="confidence-row"><div><small>Operational confidence</small><div className="confidence-track"><span style={{ width: `${confidence * 100}%` }} /></div></div><strong>{confidenceLabel(confidence)}</strong></div>{currentResult.modelUsed && <p className="model"><Database size={12} /> {currentResult.modelUsed}</p>}</article>
            <article className="specialist-card"><div className="card-label"><Network size={14} /> M4 routing</div><div className="selected-specialist"><span className="specialist-orb"><Radar size={17} /></span><div><small>Selected specialist</small><strong>{specialist.id} · {specialist.name}</strong><p>{specialist.reason}</p></div></div><div className="routing-line"><span>Last tool</span><strong>{actualTool?.tool?.replaceAll('_', ' ').toUpperCase() ?? specialist.id}</strong></div><div className="routing-line"><span>Tool status</span><strong>{actualTool?.status?.toUpperCase() ?? badge}</strong></div></article>
          </section>

          {scalarData.length > 0 && <section className="native-data"><div className="section-heading"><div><p className="overline">NATIVE OUTPUT</p><h3>What the specialist actually returned</h3></div><span>M4 ToolResult</span></div><div className="metric-row">{scalarData.map(([key, value]) => <div className="metric" key={key}><small>{key.replaceAll('_', ' ')}</small><strong>{displayValue(value)}</strong></div>)}</div></section>}

          <section className="evidence-section"><div className="section-heading"><div><p className="overline">05 · EVIDENCE</p><h3>Source imagery & generated artifacts</h3></div><span>{currentResult.artifactUrls?.length ?? 0} backend artifact(s)</span></div><div className="evidence-grid">{sourceImages.map((image) => <EvidenceImage key={image.label} label={image.label} src={image.src} />)}{currentResult.artifactUrls?.filter((url) => /\.(png|jpe?g|webp|tiff?)($|\?)/i.test(url)).map((url) => <EvidenceImage key={url} label="GENERATED ARTIFACT" src={url} artifact />)}</div>{currentResult.artifactUrls && currentResult.artifactUrls.length > 0 && <div className="artifact-links">{currentResult.artifactUrls.map((url) => <a href={url} target="_blank" rel="noreferrer" key={url}><Layers3 size={13} /> {url.split('/').pop()} <ArrowUpRight size={12} /></a>)}</div>}{currentResult.evidence.length > 0 && <div className="evidence-list">{currentResult.evidence.slice(0, 6).map((item) => <div key={item.id}><span>{item.label}</span><small>{item.description}</small><strong>{confidenceLabel(item.confidence)}</strong></div>)}</div>}</section>

          <section className="geo-section"><div className="section-heading"><div><p className="overline">06 · GEOSPATIAL</p><h3>Geospatial evidence</h3></div><span>{mapArtifact || externalMapUrl ? 'AVAILABLE' : 'NOT RETURNED'}</span></div>{mapArtifact ? <div className="map-wrap"><iframe src={mapArtifact} title="M5 generated geospatial evidence" loading="lazy" /></div> : externalMapUrl ? <div className="map-wrap"><iframe src={externalMapUrl} title="Geospatial evidence map" loading="lazy" /></div> : <div className="geo-empty"><MapPin size={19} /><div><strong>No geospatial reference was returned.</strong><p>This is intentionally not simulated. Ask for “where” or “regions”, or upload a georeferenced GeoTIFF so M4 can invoke grounding + M5 GIS.</p></div></div>}{coordinates && <div className="geo-meta"><span><MapPin size={12} /> {coordinates.locationName ?? 'Analysis location'}</span><strong>{coordinates.lat.toFixed(5)}, {coordinates.lng.toFixed(5)}</strong><small>{coordinates.crs ?? 'EPSG:4326'}</small>{geoJsonArtifact && <a href={geoJsonArtifact} target="_blank" rel="noreferrer">GeoJSON <ArrowUpRight size={11} /></a>}</div>}</section>

          <section className="trace-section"><div className="section-heading"><div><p className="overline">07 · PROVENANCE</p><h3>Execution trace</h3></div></div><div className="trace-list">{trace.map((step) => <div key={step}><Check size={14} /> {step}</div>)}</div></section>
        </>}

        <section className="specialist-status"><div><p className="overline">SYSTEM</p><h3>Specialist status</h3></div><div className="specialist-list">{SPECIALISTS.map((item) => <div className={item.id === specialist.id ? 'selected' : ''} key={item.id}><span><i />{item.id}</span><strong>{item.name}</strong><small>{item.role}</small></div>)}</div></section>
        {history.length > 0 && <div className="history-note"><Activity size={13} /> {history.length} analysis run{history.length === 1 ? '' : 's'} this session · API {API_BASE_URL}</div>}
      </main>
    </div>
  );
};

interface UploadTileProps { label: string; slot: SlotState; onFile: (file: File) => void; }

const UploadTile: React.FC<UploadTileProps> = ({ label, slot, onFile }) => {
  const inputId = `upload-${label.replace(/\s+/g, '-').toLowerCase()}`;
  const [dragging, setDragging] = useState(false);
  return <div className={`upload-tile ${slot.previewUrl ? 'filled' : ''} ${dragging ? 'dragging' : ''}`} onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); onFile(event.dataTransfer.files[0]); }}><input id={inputId} type="file" accept="image/png,image/jpeg,image/tiff,.tif,.tiff" onChange={(event) => onFile(event.target.files?.[0] as File)} />{slot.previewUrl ? <><img src={slot.previewUrl} alt={label} /><div className="upload-caption"><span>{label}</span><label htmlFor={inputId}>Replace</label></div></> : <label className="upload-empty" htmlFor={inputId}><Upload size={20} /><strong>{label}</strong><span>Drop image or browse</span><small>JPG · PNG · GeoTIFF</small></label>}</div>;
};

interface EvidenceImageProps { label: string; src?: string; artifact?: boolean; }
const EvidenceImage: React.FC<EvidenceImageProps> = ({ label, src, artifact }) => <figure className={`evidence-image ${artifact ? 'artifact' : ''}`}><figcaption><span>{label}</span>{artifact && <small>BACKEND</small>}</figcaption>{src ? <img src={src} alt={label} /> : <div className="missing-image"><FileImage size={18} /><span>No image</span></div>}</figure>;

export default App;
