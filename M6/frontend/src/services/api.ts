import type { AnalysisResponse, ToolResultRecord, ViewMode } from '../types';
import { SAMPLE_SCENARIOS } from './mockData';

export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Demo-only bi-temporal construction scenario.
 *
 * This changes only the frontend demonstration dataset. It does not alter
 * M1/M2/M3 models or the live backend path.
 */
const SVG_CONSTRUCTION_BEFORE = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="800" height="600">
  <defs>
    <linearGradient id="terrainBefore" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="%235b654d"/>
      <stop offset="55%" stop-color="%23817d63"/>
      <stop offset="100%" stop-color="%23a89a75"/>
    </linearGradient>
    <pattern id="fieldsBefore" width="90" height="90" patternUnits="userSpaceOnUse">
      <rect width="90" height="90" fill="%237d805f"/>
      <path d="M0 75 L90 20 M-20 55 L45 -5 M45 95 L105 35" stroke="%23949770" stroke-width="5" opacity="0.65"/>
    </pattern>
  </defs>
  <rect width="800" height="600" fill="url(%23terrainBefore)"/>
  <rect x="0" y="330" width="800" height="270" fill="url(%23fieldsBefore)" opacity="0.82"/>
  <path d="M0 510 L800 395" stroke="%23c8b98a" stroke-width="18" opacity="0.75"/>
  <path d="M95 0 L120 600" stroke="%23b9ad82" stroke-width="12" opacity="0.65"/>
  <path d="M180 120 L390 155 L360 255 L150 225 Z" fill="%23656b63" stroke="%23a9aa99" stroke-width="4"/>
  <path d="M205 145 L370 172 L345 225 L185 202 Z" fill="%237f857b"/>
  <path d="M455 205 L610 220 L595 305 L440 290 Z" fill="%236e756c" stroke="%23a9aa99" stroke-width="4"/>
  <path d="M480 225 L585 235 L575 280 L465 270 Z" fill="%238a8f84"/>
  <rect x="585" y="345" width="92" height="56" fill="%2374776f" stroke="%23b5b19b" stroke-width="3"/>
  <text x="24" y="36" fill="%23f8fafc" font-family="monospace" font-size="14" font-weight="bold">SENTINEL-2 | T1 | EXISTING BUILT-UP FOOTPRINT</text>
  <text x="24" y="572" fill="%23facc15" font-family="sans-serif" font-size="13">BASELINE: Established settlement with open parcels at the eastern edge</text>
</svg>`;

const SVG_CONSTRUCTION_AFTER = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="800" height="600">
  <defs>
    <linearGradient id="terrainAfter" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="%235b654d"/>
      <stop offset="55%" stop-color="%23817d63"/>
      <stop offset="100%" stop-color="%23a89a75"/>
    </linearGradient>
    <pattern id="fieldsAfter" width="90" height="90" patternUnits="userSpaceOnUse">
      <rect width="90" height="90" fill="%237d805f"/>
      <path d="M0 75 L90 20 M-20 55 L45 -5 M45 95 L105 35" stroke="%23949770" stroke-width="5" opacity="0.65"/>
    </pattern>
  </defs>
  <rect width="800" height="600" fill="url(%23terrainAfter)"/>
  <rect x="0" y="330" width="800" height="270" fill="url(%23fieldsAfter)" opacity="0.82"/>
  <path d="M0 510 L800 395" stroke="%23c8b98a" stroke-width="18" opacity="0.75"/>
  <path d="M95 0 L120 600" stroke="%23b9ad82" stroke-width="12" opacity="0.65"/>
  <path d="M180 120 L390 155 L360 255 L150 225 Z" fill="%23656b63" stroke="%23a9aa99" stroke-width="4"/>
  <path d="M205 145 L370 172 L345 225 L185 202 Z" fill="%237f857b"/>
  <path d="M455 205 L610 220 L595 305 L440 290 Z" fill="%236e756c" stroke="%23a9aa99" stroke-width="4"/>
  <path d="M480 225 L585 235 L575 280 L465 270 Z" fill="%238a8f84"/>
  <rect x="600" y="110" width="145" height="105" rx="4" fill="%236d736b" stroke="%23d2d1c2" stroke-width="4"/>
  <rect x="620" y="125" width="48" height="32" fill="%23969a8d"/>
  <rect x="680" y="125" width="48" height="32" fill="%23969a8d"/>
  <rect x="620" y="168" width="48" height="32" fill="%23969a8d"/>
  <rect x="680" y="168" width="48" height="32" fill="%23969a8d"/>
  <rect x="555" y="300" width="160" height="92" rx="4" fill="%23727870" stroke="%23d2d1c2" stroke-width="4"/>
  <rect x="575" y="315" width="52" height="27" fill="%23969a8d"/>
  <rect x="642" y="315" width="52" height="27" fill="%23969a8d"/>
  <rect x="575" y="351" width="52" height="27" fill="%23969a8d"/>
  <rect x="642" y="351" width="52" height="27" fill="%23969a8d"/>
  <path d="M585 90 L585 420 M535 250 L760 250" stroke="%23d4c8a0" stroke-width="14"/>
  <text x="24" y="36" fill="%23f8fafc" font-family="monospace" font-size="14" font-weight="bold">SENTINEL-2 | T2 | EXPANDED BUILT-UP FOOTPRINT</text>
  <text x="24" y="572" fill="%23facc15" font-family="sans-serif" font-size="13">OBSERVATION: New rectangular structures and access roads appear in previously open parcels</text>
</svg>`;

const SVG_CONSTRUCTION_OVERLAY = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="800" height="600">
  <rect width="800" height="600" fill="rgba(0,0,0,0)"/>
  <rect x="600" y="110" width="145" height="105" rx="5" fill="rgba(34,211,238,0.32)" stroke="%2322d3ee" stroke-width="5"/>
  <rect x="555" y="300" width="160" height="92" rx="5" fill="rgba(34,211,238,0.32)" stroke="%2322d3ee" stroke-width="5"/>
  <path d="M585 90 L585 420 M535 250 L760 250" stroke="%23f59e0b" stroke-width="8" stroke-dasharray="12,8"/>
  <text x="570" y="455" fill="%23ffffff" font-family="sans-serif" font-size="16" font-weight="bold">NEW CONSTRUCTION</text>
  <text x="570" y="478" fill="%23fef08a" font-family="sans-serif" font-size="14">~18.6 ha detected change</text>
</svg>`;

function applyConstructionDemoScenario(): void {
  const scenario = SAMPLE_SCENARIOS.find((item) => item.id === 'change-detection-flood');
  if (!scenario) return;

  scenario.title = 'New Construction & Built-Up Area Change';
  scenario.query = 'Identify newly constructed areas.';
  scenario.description = 'Bi-temporal comparison of satellite observations highlighting newly constructed structures, access roads, and built-up area expansion.';
  scenario.images = {
    before: SVG_CONSTRUCTION_BEFORE,
    after: SVG_CONSTRUCTION_AFTER,
  };

  scenario.mockResponse = {
    ...scenario.mockResponse,
    id: 'res-construction-demo-001',
    mode: 'change-detection',
    query: 'Identify newly constructed areas.',
    answer:
      'Bi-temporal comparison shows clear expansion of the built-up footprint between T1 and T2. Newly constructed rectangular structures are concentrated along the eastern edge of the existing settlement, with associated access-road expansion into previously open parcels. The demo change footprint is approximately 18.6 hectares; this represents a construction-area change, not flooding.',
    resultSummary:
      'Detected approximately 18.6 ha of newly constructed built-up area across two concentrated development zones.',
    confidence: 0.91,
    modelUsed: 'SatQuery Demo Change Analysis (M2/M4 Pipeline)',
    executionTimeMs: 1180,
    status: 'completed',
    beforeImageUrl: SVG_CONSTRUCTION_BEFORE,
    afterImageUrl: SVG_CONSTRUCTION_AFTER,
    overlayImageUrl: SVG_CONSTRUCTION_OVERLAY,
    changes: [
      {
        id: 'change-construction-1',
        type: 'construction',
        severity: 'high',
        estimatedAreaHa: 18.6,
        description: 'New rectangular building footprints detected in previously open parcels along the eastern settlement edge.',
      },
      {
        id: 'change-road-1',
        type: 'infrastructure',
        severity: 'moderate',
        estimatedAreaHa: 3.2,
        description: 'New linear access-road connections appear around the expanded construction footprint.',
      },
    ],
    reasoningSteps: [
      'M4 Router classified the request as bi-temporal construction/change analysis and routed it to M2.',
      'T1 and T2 observations were compared on the same spatial reference frame.',
      'Persistent high-change regions were grouped into connected construction footprints.',
      'New structures and associated access roads were identified as the dominant change class.',
      'Geospatial evidence and area estimates were synthesized for the frontend result.',
    ],
    evidence: [
      {
        id: 'construction-1',
        label: 'New construction zone A',
        confidence: 0.93,
        category: 'change',
        description: 'Dense cluster of new rectangular structures in the eastern parcel.',
        bbox: [110, 600, 215, 745],
      },
      {
        id: 'construction-2',
        label: 'New construction zone B',
        confidence: 0.89,
        category: 'change',
        description: 'Secondary built-up expansion with regular building spacing and access-road connection.',
        bbox: [300, 555, 392, 715],
      },
      {
        id: 'construction-3',
        label: 'Access-road expansion',
        confidence: 0.87,
        category: 'infrastructure',
        description: 'Linear transport corridor connecting the new development parcels.',
        bbox: [250, 535, 420, 760],
      },
    ],
    coordinates: {
      lat: 28.6139,
      lng: 77.209,
      zoom: 12,
      locationName: 'Delhi NCR Urban Expansion Demo',
      crs: 'EPSG:4326',
    },
    boundingBox: {
      north: 28.72,
      south: 28.52,
      east: 77.31,
      west: 77.10,
    },
    sensorMetadata: {
      platform: 'Sentinel-2',
      sensor: 'MultiSpectral Instrument Level-2A',
      gsd: '10m / pixel',
      crs: 'EPSG:4326 (WGS 84)',
      acquisitionDate: '2026-08-04 UTC',
      bands: ['B02-Blue', 'B03-Green', 'B04-Red', 'B08-NIR'],
    },
    geojson: [
      {
        type: 'Feature',
        properties: { class: 'new_construction', area_ha: 18.6, confidence: 0.91 },
        geometry: {
          type: 'Polygon',
          coordinates: [[[77.23, 28.59], [77.29, 28.59], [77.29, 28.66], [77.23, 28.66], [77.23, 28.59]]],
        },
      },
    ],
  };
}

applyConstructionDemoScenario();

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
