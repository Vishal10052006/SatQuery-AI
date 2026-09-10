import type { AnalysisResponse, ViewMode } from '../types';
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

