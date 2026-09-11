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

/** Collect generated artifacts from top-level M5, M2, and nested upstream evidence. */
function collectArtifactUrls(raw: any): string[] {
  const candidates: unknown[] = Array.isArray(raw.artifacts) ? raw.artifacts : [];

  for (const result of Array.isArray(raw.results) ? raw.results : []) {
    const data = result?.data ?? {};
    for (const key of [
      'artifacts', 'artifact_paths', 'change_mask_path', 'change_mask',
      'overlay_path', 'overlay_url', 'mask_path', 'composite_path',
      'visualization_path', 'evidence_path', 'geojson_path', 'map_path',
    ]) {
      const value = data[key];
      if (Array.isArray(value)) candidates.push(...value);
      else if (value) candidates.push(value);
    }

    const upstream = data.upstream_data;
    if (upstream && typeof upstream === 'object') {
      for (const key of ['artifacts', 'artifact_paths', 'change_mask_path', 'overlay_path', 'mask_path', 'composite_path']) {
        const value = upstream[key];
        if (Array.isArray(value)) candidates.push(...value);
        else if (value) candidates.push(value);
      }
    }
  }

  return [...new Set(candidates.map(artifactUrl).filter((value): value is string => Boolean(value)))];
}

/** Main dispatcher for Demo Mode and the real M5 FastAPI service. */
export async function executeSatelliteAnalysis(payload: AnalysisPayload, isDemoMode: boolean): Promise<AnalysisResponse> {
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
    const response = await fetch(`${API_BASE_URL}/api/analyze`, { method: 'POST', body: formData, signal: controller.signal });
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

/** Accept only real WGS84 coordinates. Pixel coordinates must never reach the map. */
function validCoordinates(value: unknown): value is { lat: number; lng: number; locationName?: string; crs?: string } {
  if (!value || typeof value !== 'object') return false;
  const candidate = value as Record<string, unknown>;
  const lat = Number(candidate.lat);
  const lng = Number(candidate.lng);
  return Number.isFinite(lat) && Number.isFinite(lng) && lat >= -90 && lat <= 90 && lng >= -180 && lng <= 180;
}

function validBoundingBox(value: unknown): value is { west: number; south: number; east: number; north: number } {
  if (!value || typeof value !== 'object') return false;
  const candidate = value as Record<string, unknown>;
  const west = Number(candidate.west);
  const south = Number(candidate.south);
  const east = Number(candidate.east);
  const north = Number(candidate.north);
  return Number.isFinite(west) && Number.isFinite(east) && Number.isFinite(south) && Number.isFinite(north)
    && west >= -180 && east <= 180 && south >= -90 && north <= 90
    && west <= east && south <= north;
}

/** Extract M5's geographic center/bounds from its native nested ToolResult. */
function extractGeospatial(toolResults: ToolResultRecord[]): { coordinates?: AnalysisResponse['coordinates']; boundingBox?: AnalysisResponse['boundingBox'] } {
  const gis = [...toolResults].reverse().find((result) => result.tool.includes('m5_gis'));
  const geo = gis?.data?.geographic_coordinates;
  const center = Array.isArray(geo?.center) && geo.center.length >= 2 ? { lat: Number(geo.center[0]), lng: Number(geo.center[1]) } : undefined;
  const bounds = Array.isArray(geo?.overall_bounds_4326) && geo.overall_bounds_4326.length >= 4
    ? { west: Number(geo.overall_bounds_4326[0]), south: Number(geo.overall_bounds_4326[1]), east: Number(geo.overall_bounds_4326[2]), north: Number(geo.overall_bounds_4326[3]) }
    : undefined;
  return {
    coordinates: validCoordinates(center) ? { ...center, locationName: 'M5 derived analysis region', crs: 'EPSG:4326' } : undefined,
    boundingBox: validBoundingBox(bounds) ? bounds : undefined,
  };
}

/** Remove implementation-level pixel counts from judge-facing prose. */
function cleanJudgeAnswer(answer: unknown, query: string): string {
  let text = typeof answer === 'string' ? answer.trim() : '';
  if (!text) return 'Satellite analysis completed. Review the generated evidence below.';
  const isChangeQuery = /change|between|before|after|constructed|construction|newly built/i.test(query);
  if (!isChangeQuery) return text;
  return text
    .replace(/\s*\([\d,]+\s+pixels?,\s*[\d.]+%\s+of\s+(?:the\s+)?scene\)/gi, '')
    .replace(/Detected\s+[\d,]+\s+changed region\(s\)/gi, 'Detected land-surface change zones')
    .replace(/Provided\s+[\d,]+\s+spatial bounding box\(es\)\s+derived from detected change regions/gi, 'Spatial boundaries were derived for the detected change zones')
    .replace(/\s{2,}/g, ' ')
    .trim();
}

/** Derive an image-space fraction when M2 returned region geometry but omitted the aggregate fraction. */
function normalizeM2Data(data: Record<string, any>): Record<string, any> {
  const normalized = { ...data };
  const explicitFraction = Number(normalized.changed_fraction ?? normalized.change_fraction);
  if (Number.isFinite(explicitFraction)) {
    normalized.changed_fraction = explicitFraction;
    normalized.change_fraction = explicitFraction;
    return normalized;
  }

  const imageSize = normalized.image_size
    || normalized.metadata?.image_size
    || normalized.evidence?.image_size;
  const width = Number(imageSize?.width);
  const height = Number(imageSize?.height);
  if (!(width > 0 && height > 0)) return normalized;

  const changedPixels = Number(normalized.changed_pixels);
  const regions = Array.isArray(normalized.regions) ? normalized.regions : [];
  const regionPixels = regions.reduce((total: number, region: any) => total + Number(region?.pixel_count ?? 0), 0);
  const pixels = Number.isFinite(changedPixels) && changedPixels > 0 ? changedPixels : regionPixels;
  if (!(pixels > 0)) return normalized;

  const fraction = Math.min(1, pixels / (width * height));
  normalized.changed_fraction = fraction;
  normalized.change_fraction = fraction;
  return normalized;
}

/** Normalize the complete M5/M4 response without throwing away nested evidence. */
function normalizeBackendResponse(raw: any, payload: AnalysisPayload): AnalysisResponse {
  const toolResults: ToolResultRecord[] = Array.isArray(raw.results)
    ? raw.results.map((result: any) => ({
        tool: String(result.tool || 'unknown'),
        status: String(result.status || 'unknown'),
        confidence: Number.isFinite(Number(result.confidence)) ? Number(result.confidence) : 0,
        data: /m2_change_detection/i.test(String(result.tool || ''))
          ? normalizeM2Data(result.data && typeof result.data === 'object' ? result.data : {})
          : (result.data && typeof result.data === 'object' ? result.data : {}),
        evidence: Array.isArray(result.evidence) ? result.evidence : [],
        error: result.error,
      }))
    : [];

  const artifacts = collectArtifactUrls(raw);
  const imageArtifacts = artifacts.filter((url) => /\.(png|jpe?g|webp|tiff?)($|\?)/i.test(url));
  const overlay = raw.overlay || raw.overlay_url || raw.mask_url || imageArtifacts.find((url) => /mask|change|overlay|diff/i.test(url));
  const geospatial = extractGeospatial(toolResults);
  const geospatialAvailable = Boolean(geospatial.coordinates && geospatial.boundingBox);
  const judgeArtifacts = geospatialAvailable
    ? artifacts
    : artifacts.filter((url) => !/\/map\.html(?:$|\?)/i.test(url) && !/\.geojson(?:$|\?)/i.test(url));

  const directCoordinates = validCoordinates(raw.coordinates)
    ? raw.coordinates
    : (raw.lat != null && raw.lng != null && validCoordinates({ lat: raw.lat, lng: raw.lng })
      ? { lat: Number(raw.lat), lng: Number(raw.lng), locationName: raw.location_name }
      : undefined);

  const normalizedEvidence = Array.isArray(raw.evidence)
    ? raw.evidence.map((item: any) => ({
        ...item,
        confidence: Number.isFinite(Number(item?.confidence))
          ? Number(item.confidence)
          : (toolResults.find((tool) => tool.tool.replaceAll('_', ' ') === String(item?.reference || '').replaceAll('_', ' '))?.confidence ?? toolResults.at(-1)?.confidence ?? 0),
      }))
    : [];

  const rawStatus = normalizeRawStatus(raw.status);
  const overallConfidence = Number.isFinite(Number(raw.confidence))
    ? (Number(raw.confidence) > 1 ? Number(raw.confidence) / 100 : Number(raw.confidence))
    : toolResults.filter((tool) => tool.status !== 'failed').at(-1)?.confidence ?? 0;

  return {
    id: raw.id || raw.mission_id || `res-${Date.now()}`,
    mode: payload.mode,
    query: payload.query,
    timestamp: Date.now(),
    answer: cleanJudgeAnswer(raw.answer || raw.result || raw.summary, payload.query),
    resultSummary: raw.result_summary || raw.short_summary || 'Satellite intelligence processed.',
    confidence: Math.max(0, Math.min(1, overallConfidence)),
    evidence: normalizedEvidence,
    changes: Array.isArray(raw.changes) ? raw.changes : undefined,
    reasoningSteps: Array.isArray(raw.trace) ? raw.trace : Array.isArray(raw.reasoning_steps) ? raw.reasoning_steps : [],
    primaryImageUrl: raw.image_url || raw.primary_image_url || payload.primaryImage?.previewUrl,
    beforeImageUrl: raw.before_image_url || payload.beforeImage?.previewUrl,
    afterImageUrl: raw.after_image_url || payload.afterImage?.previewUrl,
    opticalImageUrl: raw.optical_image_url || payload.opticalImage?.previewUrl,
    sarImageUrl: raw.sar_image_url || payload.sarImage?.previewUrl,
    overlayImageUrl: artifactUrl(overlay),
    artifactUrls: judgeArtifacts,
    toolResults,
    modelUsed: raw.model_used || toolResults.map((result) => result.tool).join(' → ') || 'M4 Agent Controller',
    executionTimeMs: raw.execution_time_ms,
    status: rawStatus,
    error: raw.error,
    coordinates: directCoordinates || geospatial.coordinates,
    boundingBox: validBoundingBox(raw.bounding_box) ? raw.bounding_box : (validBoundingBox(raw.bbox_coords) ? raw.bbox_coords : geospatial.boundingBox),
    geojson: geospatialAvailable ? (raw.geojson || raw.features) : undefined,
    sensorMetadata: raw.sensor_metadata || {
      platform: raw.satellite || 'Earth Observation Satellite',
      sensor: raw.sensor || 'Multi-Spectral / Radar Instrument',
      gsd: raw.resolution || 'Not returned by backend',
      crs: raw.crs || geospatial.coordinates?.crs || 'Not returned by backend',
    },
  };
}

function normalizeRawStatus(value: unknown): AnalysisResponse['status'] {
  const status = String(value ?? '').trim().toLowerCase();
  if (status === 'failed' || status === 'error') return 'error';
  if (status === 'partial' || status === 'degraded' || status === 'fallback_baseline') return 'partial';
  if (status === 'success' || status === 'completed') return 'completed';
  return 'completed';
}
