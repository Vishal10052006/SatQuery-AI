import type { AnalysisResponse, ToolResultRecord, ViewMode } from '../types';
import { SAMPLE_SCENARIOS } from './mockData';

export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface AnalysisPayload {
  mode: ViewMode;
  query: string;
  primaryImage?: { file?: File; previewUrl?: string };
  beforeImage?: { file?: File; previewUrl?: string };
  afterImage?: { file?: File; previewUrl?: string };
  opticalImage?: { file?: File; previewUrl?: string };
  sarImage?: { file?: File; previewUrl?: string };
}

/** Check whether the M5 FastAPI service is reachable. */
export async function checkBackendHealth(): Promise<{ isOnline: boolean; url: string; message: string }> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2500);
    const response = await fetch(`${API_BASE_URL}/api/health`, { method: 'GET', signal: controller.signal });
    clearTimeout(timeoutId);

    return response.ok
      ? { isOnline: true, url: API_BASE_URL, message: 'Connected to M5 FastAPI backend' }
      : { isOnline: false, url: API_BASE_URL, message: `Backend responded with HTTP ${response.status}` };
  } catch (error) {
    return {
      isOnline: false,
      url: API_BASE_URL,
      message: error instanceof Error && error.name === 'AbortError' ? 'Connection timed out' : 'Backend offline or unreachable',
    };
  }
}

/** Convert a backend-relative artifact path into a browser URL. */
function artifactUrl(value: unknown): string | undefined {
  if (typeof value !== 'string' || !value) return undefined;
  if (/^https?:\/\//i.test(value)) return value;
  return `${API_BASE_URL.replace(/\/$/, '')}/${value.replace(/^\//, '')}`;
}

/** Collect generated artifacts from both top-level M5 output and nested specialist results. */
function collectArtifactUrls(raw: any): string[] {
  const candidates: unknown[] = Array.isArray(raw.artifacts) ? raw.artifacts : [];

  for (const result of Array.isArray(raw.results) ? raw.results : []) {
    const data = result?.data ?? {};
    for (const key of [
      'artifacts',
      'artifact_paths',
      'change_mask_path',
      'change_mask',
      'overlay_path',
      'overlay_url',
      'mask_path',
      'visualization_path',
      'evidence_path',
      'geojson_path',
      'map_path',
    ]) {
      const value = data[key];
      if (Array.isArray(value)) candidates.push(...value);
      else if (value) candidates.push(value);
    }
  }

  return [...new Set(candidates.map(artifactUrl).filter((value): value is string => Boolean(value)))];
}

/** Main dispatcher for Demo Mode and the real M5 FastAPI service. */
export async function executeSatelliteAnalysis(
  payload: AnalysisPayload,
  isDemoMode: boolean,
): Promise<AnalysisResponse> {
  if (isDemoMode) {
    await new Promise((resolve) => setTimeout(resolve, 900));
    return getMatchingDemoResponse(payload);
  }

  const formData = new FormData();
  formData.append('query', payload.query);
  formData.append('mode', payload.mode);
  if (payload.primaryImage?.file) formData.append('image', payload.primaryImage.file);
  if (payload.beforeImage?.file) formData.append('before_image', payload.beforeImage.file);
  if (payload.afterImage?.file) formData.append('after_image', payload.afterImage.file);
  if (payload.opticalImage?.file) formData.append('optical_image', payload.opticalImage.file);
  if (payload.sarImage?.file) formData.append('sar_image', payload.sarImage.file);

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 30000);

  try {
    const response = await fetch(`${API_BASE_URL}/api/analyze`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!response.ok) {
      let detail = response.statusText;
      try {
        const body = await response.json();
        detail = body.detail || body.message || detail;
      } catch {
        // Keep HTTP status text when the backend did not return JSON.
      }
      throw new Error(`M5 Backend Error (${response.status}): ${detail}`);
    }

    return normalizeBackendResponse(await response.json(), payload);
  } catch (error) {
    clearTimeout(timeoutId);
    if (error instanceof Error && error.name === 'AbortError') throw new Error('API request timed out after 30s.');
    if (error instanceof Error && error.message.includes('Failed to fetch')) throw new Error(`Unable to reach M5 FastAPI Backend at ${API_BASE_URL}.`);
    throw error;
  }
}

/** Build a Demo response while keeping the selected source imagery visible. */
function getMatchingDemoResponse(payload: AnalysisPayload): AnalysisResponse {
  let matched = SAMPLE_SCENARIOS.find((scenario) => scenario.mode === payload.mode);
  if (!matched) matched = SAMPLE_SCENARIOS[0];

  const base = matched.mockResponse;
  return {
    ...base,
    id: `res-${Date.now()}`,
    timestamp: Date.now(),
    query: payload.query.trim() || base.query,
    primaryImageUrl: payload.primaryImage?.previewUrl || base.primaryImageUrl,
    beforeImageUrl: payload.beforeImage?.previewUrl || base.beforeImageUrl,
    afterImageUrl: payload.afterImage?.previewUrl || base.afterImageUrl,
    opticalImageUrl: payload.opticalImage?.previewUrl || base.opticalImageUrl,
    sarImageUrl: payload.sarImage?.previewUrl || base.sarImageUrl,
  };
}

/** Extract M5's geographic center/bounds from its native nested ToolResult. */
function extractGeospatial(toolResults: ToolResultRecord[]): {
  coordinates?: AnalysisResponse['coordinates'];
  boundingBox?: AnalysisResponse['boundingBox'];
} {
  const gis = [...toolResults].reverse().find((result) => result.tool.includes('m5_gis'));
  const geo = gis?.data?.geographic_coordinates;
  const center = Array.isArray(geo?.center) && geo.center.length >= 2 ? geo.center : undefined;
  const bounds = Array.isArray(geo?.overall_bounds_4326) && geo.overall_bounds_4326.length >= 4 ? geo.overall_bounds_4326 : undefined;

  return {
    coordinates: center
      ? { lat: Number(center[0]), lng: Number(center[1]), locationName: 'M5 derived analysis region', crs: 'EPSG:4326' }
      : undefined,
    boundingBox: bounds
      ? { west: Number(bounds[0]), south: Number(bounds[1]), east: Number(bounds[2]), north: Number(bounds[3]) }
      : undefined,
  };
}

/** Normalize the complete M5/M4 response without throwing away nested evidence. */
function normalizeBackendResponse(raw: any, payload: AnalysisPayload): AnalysisResponse {
  const toolResults: ToolResultRecord[] = Array.isArray(raw.results)
    ? raw.results.map((result: any) => ({
        tool: String(result.tool || 'unknown'),
        status: String(result.status || 'unknown'),
        confidence: typeof result.confidence === 'number' ? result.confidence : 0,
        data: result.data && typeof result.data === 'object' ? result.data : {},
        evidence: Array.isArray(result.evidence) ? result.evidence : [],
        error: result.error,
      }))
    : [];

  const artifacts = collectArtifactUrls(raw);
  const imageArtifacts = artifacts.filter((url) => /\.(png|jpe?g|webp|tiff?)($|\?)/i.test(url));
  const overlay = raw.overlay || raw.overlay_url || raw.mask_url || imageArtifacts.find((url) => /mask|change|overlay|diff/i.test(url));
  const geospatial = extractGeospatial(toolResults);

  return {
    id: raw.id || raw.mission_id || `res-${Date.now()}`,
    mode: payload.mode,
    query: payload.query,
    timestamp: Date.now(),
    answer: raw.answer || raw.result || raw.summary || 'Analysis successfully generated by SatQuery AI.',
    resultSummary: raw.result_summary || raw.short_summary || 'Satellite intelligence processed.',
    confidence: typeof raw.confidence === 'number' ? (raw.confidence > 1 ? raw.confidence / 100 : raw.confidence) : toolResults.at(-1)?.confidence ?? 0,
    evidence: Array.isArray(raw.evidence) ? raw.evidence : [],
    changes: Array.isArray(raw.changes) ? raw.changes : undefined,
    reasoningSteps: Array.isArray(raw.trace) ? raw.trace : Array.isArray(raw.reasoning_steps) ? raw.reasoning_steps : [],
    primaryImageUrl: raw.image_url || raw.primary_image_url || payload.primaryImage?.previewUrl,
    beforeImageUrl: raw.before_image_url || payload.beforeImage?.previewUrl,
    afterImageUrl: raw.after_image_url || payload.afterImage?.previewUrl,
    opticalImageUrl: raw.optical_image_url || payload.opticalImage?.previewUrl,
    sarImageUrl: raw.sar_image_url || payload.sarImage?.previewUrl,
    overlayImageUrl: artifactUrl(overlay),
    artifactUrls: artifacts,
    toolResults,
    modelUsed: raw.model_used || toolResults.map((result) => result.tool).join(' → ') || 'M4 Agent Controller',
    executionTimeMs: raw.execution_time_ms,
    status: raw.status === 'failed' ? 'error' : 'completed',
    error: raw.error,
    coordinates: raw.coordinates || (raw.lat != null && raw.lng != null ? { lat: raw.lat, lng: raw.lng, locationName: raw.location_name } : geospatial.coordinates),
    boundingBox: raw.bounding_box || raw.bbox_coords || geospatial.boundingBox,
    geojson: raw.geojson || raw.features,
    sensorMetadata: raw.sensor_metadata || {
      platform: raw.satellite || 'Earth Observation Satellite',
      sensor: raw.sensor || 'Multi-Spectral / Radar Instrument',
      gsd: raw.resolution || 'Not returned by backend',
      crs: raw.crs || geospatial.coordinates?.crs || 'Not returned by backend',
    },
  };
}
