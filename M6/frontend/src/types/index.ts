export type ViewMode =
  | 'dashboard'
  | 'image-understanding'
  | 'change-detection'
  | 'optical-sar'
  | 'history';

export type PipelineStage =
  | 'idle'
  | 'routing'
  | 'analyzing'
  | 'evidence'
  | 'completed'
  | 'error';

export interface ImageSlot {
  id: string;
  name: string;
  url: string;
  file?: File;
  previewUrl: string;
  sizeBytes?: number;
  sensor?: string;
  acquisitionDate?: string;
}

export interface EvidenceItem {
  id: string;
  label: string;
  confidence: number;
  category: 'urban' | 'water' | 'vegetation' | 'infrastructure' | 'change' | 'radar_backscatter';
  description: string;
  bbox?: [number, number, number, number];
}

export interface DetectedChange {
  id: string;
  type: string;
  severity: 'low' | 'moderate' | 'high';
  estimatedAreaHa: number;
  description: string;
}

export interface GisCoordinates {
  lat: number;
  lng: number;
  zoom?: number;
  locationName?: string;
  crs?: string;
}

export interface GisBoundingBox {
  north: number;
  south: number;
  east: number;
  west: number;
}

export interface GisGeoJsonFeature {
  type: string;
  geometry: {
    type: string;
    coordinates: any;
  };
  properties?: Record<string, any>;
}

export interface SensorMetadata {
  platform: string;
  sensor: string;
  gsd: string;
  crs: string;
  acquisitionDate?: string;
  bands?: string[];
}

export interface ToolEvidenceRecord {
  type: string;
  reference?: string;
  description?: string;
  metadata?: Record<string, any>;
}

export interface ToolResultRecord {
  tool: string;
  status: string;
  confidence: number;
  data: Record<string, any>;
  evidence: ToolEvidenceRecord[];
  error?: string;
}

export interface AnalysisResponse {
  id: string;
  mode: ViewMode;
  query: string;
  timestamp: number;
  answer: string;
  resultSummary: string;
  confidence: number;
  evidence: EvidenceItem[];
  changes?: DetectedChange[];
  reasoningSteps: string[];
  primaryImageUrl?: string;
  beforeImageUrl?: string;
  afterImageUrl?: string;
  opticalImageUrl?: string;
  sarImageUrl?: string;
  overlayImageUrl?: string;
  artifactUrls?: string[];
  toolResults?: ToolResultRecord[];
  executionTimeMs?: number;
  modelUsed?: string;
  /** completed = all requested work operationally completed; partial = usable evidence with a degraded stage. */
  status: 'completed' | 'partial' | 'error';
  error?: string;
  coordinates?: GisCoordinates;
  boundingBox?: GisBoundingBox;
  geojson?: GisGeoJsonFeature | GisGeoJsonFeature[];
  sensorMetadata?: SensorMetadata;
}

export interface SampleScenario {
  id: string;
  mode: ViewMode;
  title: string;
  query: string;
  description: string;
  images: {
    primary?: string;
    before?: string;
    after?: string;
    optical?: string;
    sar?: string;
  };
  mockResponse: AnalysisResponse;
}
