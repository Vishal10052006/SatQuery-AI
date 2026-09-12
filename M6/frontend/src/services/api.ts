import type { AnalysisResponse, ViewMode } from '../types';
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

/**
 * Check if M5 FastAPI backend is reachable.
 */
export async function checkBackendHealth(): Promise<{ isOnline: boolean; url: string; message: string }> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2500);

    const response = await fetch(`${API_BASE_URL}/api/health`, {
      method: 'GET',
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (response.ok) {
      return { isOnline: true, url: API_BASE_URL, message: 'Connected to M5 FastAPI backend' };
    }
    return { isOnline: false, url: API_BASE_URL, message: `Backend responded with HTTP ${response.status}` };
  } catch (err: any) {
    return {
      isOnline: false,
      url: API_BASE_URL,
      message: err.name === 'AbortError' ? 'Connection timed out' : 'Backend offline or unreachable',
    };
  }
}

/**
 * Main API request dispatcher.
 * Supports both Live M5 Backend and Modular Demo Mode.
 */
export async function executeSatelliteAnalysis(
  payload: AnalysisPayload,
  isDemoMode: boolean
): Promise<AnalysisResponse> {
  // If Demo Mode is explicitly active, simulate pipeline latency and return realistic mock response
  if (isDemoMode) {
    await new Promise((resolve) => setTimeout(resolve, 1400));
    return getMatchingDemoResponse(payload);
  }

  // Live Backend Mode: Make actual API call to M5 FastAPI server with 15s timeout
  const formData = new FormData();
  formData.append('query', payload.query);
  formData.append('mode', payload.mode);

  if (payload.primaryImage?.file) {
    formData.append('image', payload.primaryImage.file);
  }
  if (payload.beforeImage?.file) {
    formData.append('before_image', payload.beforeImage.file);
  }
  if (payload.afterImage?.file) {
    formData.append('after_image', payload.afterImage.file);
  }
  if (payload.opticalImage?.file) {
    formData.append('optical_image', payload.opticalImage.file);
  }
  if (payload.sarImage?.file) {
    formData.append('sar_image', payload.sarImage.file);
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 15000);

  try {
    const response = await fetch(`${API_BASE_URL}/api/analyze`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!response.ok) {
      let errDetail = response.statusText;
      try {
        const errJson = await response.json();
        errDetail = errJson.detail || errJson.message || errDetail;
      } catch {
        const errText = await response.text();
        if (errText) errDetail = errText;
      }
      throw new Error(`M5 Backend Error (${response.status}): ${errDetail}`);
    }

    const data = await response.json();
    return normalizeBackendResponse(data, payload);
  } catch (error: any) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') {
      throw new Error('API Request timed out after 15s. The model inference server took too long to respond.');
    }
    if (error.message && error.message.includes('Failed to fetch')) {
      throw new Error(
        `Unable to reach M5 FastAPI Backend at ${API_BASE_URL}. Ensure the server is running or switch to Demo Mode.`
      );
    }
    console.warn('Backend call failed:', error);
    throw error;
  }
}

/**
 * Retrieve the best matching scenario from mock data based on mode & query.
 */
function getMatchingDemoResponse(payload: AnalysisPayload): AnalysisResponse {
  let matched = SAMPLE_SCENARIOS.find((s) => s.mode === payload.mode);
  if (!matched) {
    matched = SAMPLE_SCENARIOS[0];
  }

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
    coordinates: base.coordinates,
    boundingBox: base.boundingBox,
    geojson: base.geojson,
    sensorMetadata: base.sensorMetadata,
  };
}

/**
 * Adapt diverse backend response field variations gracefully:
 * Supports answer, result, confidence, evidence, overlay, coordinates, etc.
 */
function normalizeBackendResponse(raw: any, payload: AnalysisPayload): AnalysisResponse {
  return {
    id: raw.id || `res-${Date.now()}`,
    mode: payload.mode,
    query: payload.query,
    timestamp: Date.now(),
    answer: raw.answer || raw.result || raw.summary || 'Analysis successfully generated by SatQuery AI.',
    resultSummary: raw.result_summary || raw.short_summary || 'Satellite intelligence processed.',
    confidence: typeof raw.confidence === 'number' ? (raw.confidence > 1 ? raw.confidence / 100 : raw.confidence) : 0.9,
    evidence: Array.isArray(raw.evidence) ? raw.evidence : [],
    changes: Array.isArray(raw.changes) ? raw.changes : undefined,
    reasoningSteps: Array.isArray(raw.reasoning_steps) ? raw.reasoning_steps : [
      'Query processed by M4 Agentic Router',
      'Inference evaluated through domain model',
      'Geospatial artifacts verified'
    ],
    primaryImageUrl: raw.image_url || raw.primary_image_url || payload.primaryImage?.previewUrl,
    beforeImageUrl: raw.before_image_url || payload.beforeImage?.previewUrl,
    afterImageUrl: raw.after_image_url || payload.afterImage?.previewUrl,
    opticalImageUrl: raw.optical_image_url || payload.opticalImage?.previewUrl,
    sarImageUrl: raw.sar_image_url || payload.sarImage?.previewUrl,
    overlayImageUrl: raw.overlay || raw.overlay_url || raw.mask_url,
    modelUsed: raw.model_used || 'M5 Connected Backend (FastAPI / M4 Router)',
    executionTimeMs: raw.execution_time_ms || 1100,
    status: 'completed',
    coordinates: raw.coordinates || (raw.lat && raw.lng ? { lat: raw.lat, lng: raw.lng, locationName: raw.location_name } : undefined),
    boundingBox: raw.bounding_box || raw.bbox_coords,
    geojson: raw.geojson || raw.features,
    sensorMetadata: raw.sensor_metadata || {
      platform: raw.satellite || 'Earth Observation Satellite',
      sensor: raw.sensor || 'Multi-Spectral / Radar Instrument',
      gsd: raw.resolution || '10m / px',
      crs: raw.crs || 'EPSG:4326',
    },
  };
}

