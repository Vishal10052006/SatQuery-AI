import type { SampleScenario } from '../types';

// Standalone self-contained realistic satellite SVG renders for offline SIH demonstration
const SVG_LAND_USE = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="800" height="600">
  <defs>
    <radialGradient id="ocean" cx="20%" cy="30%" r="80%"><stop offset="0%" stop-color="%230c4a6e"/><stop offset="100%" stop-color="%23082f49"/></radialGradient>
    <pattern id="urban-grid" width="24" height="24" patternUnits="userSpaceOnUse">
      <rect width="24" height="24" fill="%23334155"/>
      <rect x="2" y="2" width="9" height="9" fill="%2364748b"/>
      <rect x="13" y="2" width="9" height="9" fill="%23475569"/>
      <rect x="2" y="13" width="9" height="9" fill="%2394a3b8"/>
      <rect x="13" y="13" width="9" height="9" fill="%23cbd5e1"/>
    </pattern>
    <pattern id="fields" width="60" height="60" patternUnits="userSpaceOnUse">
      <rect width="60" height="60" fill="%23166534"/>
      <polygon points="0,0 60,10 60,60 0,50" fill="%2315803d"/>
      <line x1="0" y1="30" x2="60" y2="35" stroke="%2314532d" stroke-width="2"/>
    </pattern>
  </defs>
  <rect width="800" height="600" fill="url(%23fields)"/>
  <path d="M0,0 Q350,180 500,600 L0,600 Z" fill="url(%23ocean)"/>
  <rect x="380" y="80" width="380" height="340" rx="12" fill="url(%23urban-grid)" stroke="%231e293b" stroke-width="4"/>
  <path d="M500,200 L800,220 L780,260 L480,240 Z" fill="%230284c7" stroke="%2338bdf8" stroke-width="2"/>
  <text x="30" y="40" fill="%23f8fafc" font-family="monospace" font-size="14" font-weight="bold">SENTINEL-2 MSI | L2A RGB | RESOLUTION: 10M/PX</text>
</svg>`;

const SVG_LAND_USE_OVERLAY = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="800" height="600">
  <!-- Coastal Water Body -->
  <path d="M0,0 Q350,180 500,600 L0,600 Z" fill="rgba(6, 182, 212, 0.35)" stroke="%2306b6d4" stroke-width="3" stroke-dasharray="6,4"/>
  <!-- High Density Urban Fabric -->
  <rect x="380" y="80" width="380" height="340" rx="8" fill="rgba(244, 63, 94, 0.28)" stroke="%23f43f5e" stroke-width="3"/>
  <!-- Agricultural Matrix -->
  <polygon points="500,430 780,430 780,590 510,590" fill="rgba(16, 185, 129, 0.3)" stroke="%2310b981" stroke-width="3"/>
  <!-- Industrial Canal -->
  <path d="M500,200 L800,220 L780,260 L480,240 Z" fill="rgba(245, 158, 11, 0.4)" stroke="%23f59e0b" stroke-width="2"/>
  <text x="520" y="120" fill="%23ffffff" font-family="sans-serif" font-weight="bold" font-size="14" filter="drop-shadow(0 2px 4px %23000)">[Class: High Density Urban]</text>
  <text x="80" y="320" fill="%23ffffff" font-family="sans-serif" font-weight="bold" font-size="14" filter="drop-shadow(0 2px 4px %23000)">[Class: Open Water Estuary]</text>
  <text x="540" y="520" fill="%23ffffff" font-family="sans-serif" font-weight="bold" font-size="14" filter="drop-shadow(0 2px 4px %23000)">[Class: Vegetated Cropland]</text>
</svg>`;

const SVG_BEFORE_CHANGE = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="800" height="600">
  <rect width="800" height="600" fill="%2315803d"/>
  <!-- River Basin Pre-event -->
  <path d="M-20,250 C240,220 380,310 820,270 L820,330 C400,380 220,290 -20,320 Z" fill="%230369a1"/>
  <circle cx="280" cy="180" r="70" fill="%23166534" stroke="%2322c55e" stroke-width="2"/>
  <rect x="420" y="380" width="220" height="140" fill="%23475569" stroke="%2364748b" stroke-width="3"/>
  <text x="30" y="40" fill="%23f8fafc" font-family="monospace" font-size="14" font-weight="bold">TIMESTAMP T1 (BEFORE): 2025-11-12 | LEVEL-2A OPTICAL</text>
  <text x="30" y="570" fill="%23facc15" font-family="sans-serif" font-size="13">STATUS: Normal Baseline (Dry Season Conditions)</text>
</svg>`;

const SVG_AFTER_CHANGE = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="800" height="600">
  <rect width="800" height="600" fill="%233f3f46"/>
  <!-- River Overbank Flood Inundation -->
  <path d="M-20,160 C260,110 390,190 820,180 L820,440 C420,530 190,440 -20,420 Z" fill="%230f172a" stroke="%230284c7" stroke-width="4"/>
  <!-- Submerged Zones -->
  <circle cx="280" cy="180" r="70" fill="%231e293b" opacity="0.8"/>
  <rect x="420" y="380" width="220" height="140" fill="%231e293b" stroke="%23ef4444" stroke-width="3"/>
  <text x="30" y="40" fill="%23f8fafc" font-family="monospace" font-size="14" font-weight="bold">TIMESTAMP T2 (AFTER): 2026-08-04 | POST-MONSOON FLOOD</text>
  <text x="30" y="570" fill="%23ef4444" font-family="sans-serif" font-size="13" font-weight="bold">EVENT: Severe Inundation & Structural Encroachment</text>
</svg>`;

const SVG_CHANGE_OVERLAY = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="800" height="600">
  <!-- Highlighted Flood Zone Mask -->
  <path d="M-20,160 C260,110 390,190 820,180 L820,440 C420,530 190,440 -20,420 Z" fill="rgba(244, 63, 94, 0.45)" stroke="%23f43f5e" stroke-width="4"/>
  <rect x="420" y="380" width="220" height="140" fill="rgba(245, 158, 11, 0.5)" stroke="%23f59e0b" stroke-width="3" stroke-dasharray="6,4"/>
  <text x="250" y="300" fill="%23ffffff" font-family="sans-serif" font-weight="bold" font-size="16" filter="drop-shadow(0 2px 4px %23000)">[Detected Inundation: 142.5 Hectares]</text>
  <text x="430" y="360" fill="%23fef08a" font-family="sans-serif" font-weight="bold" font-size="14" filter="drop-shadow(0 2px 4px %23000)">[Infrastructure Submersion: 87%]</text>
</svg>`;

const SVG_OPTICAL_IMAGE = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="800" height="600">
  <!-- Coastal terrain under heavy cloud cover -->
  <rect width="800" height="600" fill="%230f766e"/>
  <path d="M400,0 L800,0 L800,600 L300,600 Z" fill="%230369a1"/>
  <!-- Heavy Cloud Bank -->
  <ellipse cx="250" cy="180" rx="220" ry="120" fill="%23e2e8f0" opacity="0.88"/>
  <ellipse cx="480" cy="240" rx="260" ry="140" fill="%23cbd5e1" opacity="0.92"/>
  <ellipse cx="320" cy="420" rx="240" ry="130" fill="%23f1f5f9" opacity="0.85"/>
  <text x="30" y="40" fill="%23f8fafc" font-family="monospace" font-size="14" font-weight="bold">SENSOR: OPTICAL (SENTINEL-2 RGB) | CLOUD COVER: 76.4%</text>
  <text x="30" y="570" fill="%23facc15" font-family="sans-serif" font-size="13">LIMITATION: Dense cumulus clouds obscuring shoreline and vessels</text>
</svg>`;

const SVG_SAR_IMAGE = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="800" height="600">
  <!-- SAR Speckle Radar Return -->
  <rect width="800" height="600" fill="%231e293b"/>
  <!-- Water low backscatter (smooth surface reflects away radar beam, looks dark) -->
  <path d="M400,0 L800,0 L800,600 L300,600 Z" fill="%23020617"/>
  <!-- Rough land high backscatter -->
  <polygon points="0,0 395,0 295,600 0,600" fill="%23475569"/>
  <!-- Metallic targets (ships in water) producing double-bounce bright radar response -->
  <circle cx="560" cy="180" r="9" fill="%23ffffff" stroke="%2338bdf8" stroke-width="4"/>
  <circle cx="630" cy="290" r="11" fill="%23ffffff" stroke="%2338bdf8" stroke-width="4"/>
  <circle cx="510" cy="410" r="8" fill="%23ffffff" stroke="%2338bdf8" stroke-width="4"/>
  <circle cx="690" cy="450" r="10" fill="%23ffffff" stroke="%2338bdf8" stroke-width="4"/>
  <text x="30" y="40" fill="%2338bdf8" font-family="monospace" font-size="14" font-weight="bold">SENSOR: SAR (SENTINEL-1 C-BAND VV+VH) | ALL-WEATHER PENETRATION</text>
  <text x="30" y="570" fill="%234ade80" font-family="sans-serif" font-size="13" font-weight="bold">ADVANTAGE: Cloud penetration reveals 4 maritime vessels & clean coastline</text>
</svg>`;

export const SAMPLE_SCENARIOS: SampleScenario[] = [
  {
    id: 'land-use-analysis',
    mode: 'image-understanding',
    title: 'Land-Use Classification & Feature Identification',
    query: 'Identify the major land-use features in this image.',
    description: 'High-resolution multispectral analysis mapping urban dense clusters, navigable water estuaries, and agricultural vegetation.',
    images: {
      primary: SVG_LAND_USE,
    },
    mockResponse: {
      id: 'res-land-use-001',
      mode: 'image-understanding',
      query: 'Identify the major land-use features in this image.',
      timestamp: Date.now(),
      answer: 'The satellite scene exhibits three primary thematic land-use classes: (1) An estuarine/marine water body spanning the western quadrant (~34.2% surface area), (2) Dense commercial & residential urban fabric in the north-eastern quadrant (~41.5%), characterized by orthogonal road grids and high building reflectance, and (3) Intensive agricultural cropland in the southern sector (~24.3%). An arterial transport/canal channel bisects the urban sector.',
      resultSummary: 'Classified 4 distinct spatial land-use regimes with 94.2% model confidence.',
      confidence: 0.942,
      modelUsed: 'SatQuery-GeoVLM-v2.4 (M1/M4 Pipeline)',
      executionTimeMs: 1240,
      status: 'completed',
      primaryImageUrl: SVG_LAND_USE,
      overlayImageUrl: SVG_LAND_USE_OVERLAY,
      reasoningSteps: [
        'M4 Router categorized user intent: "Thematic Land Cover Extraction" -> routed to M1 VLM module.',
        'Normalized Difference Water Index (NDWI) calculated for coastal zone delineation.',
        'Normalized Difference Vegetation Index (NDVI) identified active vegetative biomass in southern quadrants.',
        'Extracted urban building boundaries via spatial edge convolutional attention.',
        'Assembled structured evidence tags and synthesized multi-modal response.',
      ],
      evidence: [
        {
          id: 'ev-1',
          label: 'Dense Urban Fabric',
          confidence: 0.96,
          category: 'urban',
          description: 'High reflectance roof clusters and regularized road grid network (41.5% coverage).',
          bbox: [80, 380, 420, 760],
        },
        {
          id: 'ev-2',
          label: 'Estuarine Coastal Water',
          confidence: 0.98,
          category: 'water',
          description: 'Deep absorption spectrum across NIR and SWIR bands confirming open saline water body.',
          bbox: [0, 0, 600, 500],
        },
        {
          id: 'ev-3',
          label: 'Cultivated Agricultural Plots',
          confidence: 0.91,
          category: 'vegetation',
          description: 'High NDVI signature indicating active photosynthesizing crops with segmented field boundaries.',
          bbox: [430, 500, 590, 780],
        },
        {
          id: 'ev-4',
          label: 'Navigable Industrial Canal',
          confidence: 0.89,
          category: 'infrastructure',
          description: 'Engineered linear water corridor linking inland industrial nodes to coastal access.',
          bbox: [200, 500, 260, 800],
        },
      ],
      coordinates: {
        lat: 18.975,
        lng: 72.825,
        zoom: 12,
        locationName: 'Mumbai Coastal Estuary & Urban Harbor, MH, India',
        crs: 'EPSG:4326',
      },
      boundingBox: {
        north: 19.08,
        south: 18.88,
        east: 72.96,
        west: 72.74,
      },
      sensorMetadata: {
        platform: 'Sentinel-2A MSI',
        sensor: 'MultiSpectral Instrument Level-2A',
        gsd: '10m / pixel',
        crs: 'EPSG:4326 (WGS 84)',
        acquisitionDate: '2026-03-02 UTC',
        bands: ['B02-Blue (490nm)', 'B03-Green (560nm)', 'B04-Red (665nm)', 'B08-NIR (842nm)'],
      },
      geojson: [
        {
          type: 'Feature',
          properties: { name: 'High-Density Urban Fabric', class: 'urban', color: '#f43f5e' },
          geometry: {
            type: 'Polygon',
            coordinates: [
              [[72.81, 19.04], [72.92, 19.04], [72.92, 18.96], [72.81, 18.96], [72.81, 19.04]]
            ]
          }
        },
        {
          type: 'Feature',
          properties: { name: 'Estuarine Water Body', class: 'water', color: '#06b6d4' },
          geometry: {
            type: 'Polygon',
            coordinates: [
              [[72.75, 19.06], [72.81, 19.01], [72.83, 18.90], [72.75, 18.90], [72.75, 19.06]]
            ]
          }
        }
      ],
    },
  },
  {
    id: 'change-detection-flood',
    mode: 'change-detection',
    title: 'Multi-Temporal Disaster & Flood Impact Analysis',
    query: 'What changed between these two images?',
    description: 'Bi-temporal comparison evaluating river overbank inundation, flooded agricultural acreage, and submerged infrastructure.',
    images: {
      before: SVG_BEFORE_CHANGE,
      after: SVG_AFTER_CHANGE,
    },
    mockResponse: {
      id: 'res-change-002',
      mode: 'change-detection',
      query: 'What changed between these two images?',
      timestamp: Date.now(),
      answer: 'Bi-temporal comparative analysis reveals extensive hydrological flooding. The river basin width expanded from a nominal baseline of ~90 meters (T1) to over 380 meters (T2). A total of 142.5 hectares of previously fertile cropland have been submerged under standing water. Furthermore, the eastern commercial logistics compound experienced severe inundation (~87% area affected), with structural damage along the peripheral barrier dike.',
      resultSummary: 'Major flood event detected: 142.5 Ha inundated; critical infrastructure disruption identified.',
      confidence: 0.894,
      modelUsed: 'SatQuery-ChangeNet-DualStream (M2/M4 Pipeline)',
      executionTimeMs: 1820,
      status: 'completed',
      beforeImageUrl: SVG_BEFORE_CHANGE,
      afterImageUrl: SVG_AFTER_CHANGE,
      overlayImageUrl: SVG_CHANGE_OVERLAY,
      changes: [
        {
          id: 'ch-1',
          type: 'River Inundation & Bank Breach',
          severity: 'high',
          estimatedAreaHa: 142.5,
          description: 'Floodwater breached southern levee, submerging floodplain and agricultural acreage.',
        },
        {
          id: 'ch-2',
          type: 'Commercial Compound Inundation',
          severity: 'high',
          estimatedAreaHa: 30.8,
          description: 'Industrial facility engulfed; perimeter road access severed.',
        },
        {
          id: 'ch-3',
          type: 'Vegetation Loss from Waterlogging',
          severity: 'moderate',
          estimatedAreaHa: 68.2,
          description: 'Significant reduction in photosynthetic vigor due to prolonged standing water.',
        },
      ],
      reasoningSteps: [
        'M4 Router identified bi-temporal pair -> passed to M2 Siamese Change Detection network.',
        'Rigid geometric co-registration applied across T1 and T2 coordinate spaces.',
        'Calculated differential water index (ΔMNDWI = MNDWI_T2 - MNDWI_T1).',
        'Segmented change pixels using adaptive Otsu thresholding.',
        'Synthesized spatial change polygon overlay and quantitative metrics.',
      ],
      evidence: [
        {
          id: 'ev-ch-1',
          label: 'Primary Inundation Zone',
          confidence: 0.94,
          category: 'change',
          description: 'Spectral transition from terrestrial vegetation to low-reflectance muddy floodwater.',
        },
        {
          id: 'ev-ch-2',
          label: 'Submerged Structural Footprint',
          confidence: 0.88,
          category: 'change',
          description: 'Compound walls visible in T1 now partially occluded by water reflection in T2.',
        },
      ],
      coordinates: {
        lat: 26.182,
        lng: 91.745,
        zoom: 11,
        locationName: 'Brahmaputra River Basin Inundation Zone, Assam, India',
        crs: 'EPSG:4326',
      },
      boundingBox: {
        north: 26.32,
        south: 26.04,
        east: 91.95,
        west: 91.55,
      },
      sensorMetadata: {
        platform: 'Sentinel-2B / Landsat-9 Dual-Stream',
        sensor: 'Bi-Temporal Optical Radiometer',
        gsd: '10m / pixel',
        crs: 'EPSG:4326 (WGS 84)',
        acquisitionDate: 'T1: 2025-11-12 | T2: 2026-08-04 UTC',
        bands: ['B03-Green (560nm)', 'B08-NIR (842nm)', 'B11-SWIR (1610nm)'],
      },
      geojson: [
        {
          type: 'Feature',
          properties: { name: 'Severe Flood Inundation Zone (142.5 Ha)', class: 'flood', color: '#ef4444' },
          geometry: {
            type: 'Polygon',
            coordinates: [
              [[91.62, 26.24], [91.85, 26.26], [91.88, 26.14], [91.65, 26.12], [91.62, 26.24]]
            ]
          }
        }
      ],
    },
  },
  {
    id: 'optical-sar-comparison',
    mode: 'optical-sar',
    title: 'Multi-Sensor Fusion: Optical vs SAR Penetration',
    query: 'Compare the optical and SAR imagery.',
    description: 'Complementary sensor synergy demonstrating Synthetic Aperture Radar (SAR) cloud penetration and metallic vessel backscatter detection.',
    images: {
      optical: SVG_OPTICAL_IMAGE,
      sar: SVG_SAR_IMAGE,
    },
    mockResponse: {
      id: 'res-sar-003',
      mode: 'optical-sar',
      query: 'Compare the optical and SAR imagery.',
      timestamp: Date.now(),
      answer: 'Multi-sensor inspection highlights the distinctive physical sensing mechanisms of Optical vs Synthetic Aperture Radar (SAR). While the optical imagery is 76.4% obscured by dense cumulus cloud formations, the C-Band SAR instrument effortlessly penetrates the atmospheric interference. The smooth open water surface acts as a specular reflector, appearing dark in SAR, which generates high-contrast "double-bounce" radar returns identifying 4 maritime vessels that are completely invisible in the optical band.',
      resultSummary: 'SAR penetrates 76.4% cloud cover, uncovering 4 maritime vessels and exact shoreline.',
      confidence: 0.918,
      modelUsed: 'SatQuery-MultiSensor-CrossFusion (M3/M4 Pipeline)',
      executionTimeMs: 1490,
      status: 'completed',
      opticalImageUrl: SVG_OPTICAL_IMAGE,
      sarImageUrl: SVG_SAR_IMAGE,
      reasoningSteps: [
        'M4 Agentic Router detected multimodal sensor pair: Sentinel-2 MSI + Sentinel-1 SAR.',
        'Analyzed optical band cloud masking via Sentinel-2 Scene Classification Layer (SCL).',
        'Calibrated Sentinel-1 SAR backscatter coefficient (Sigma-0 in dB) for VV and VH polarizations.',
        'Applied Constant False Alarm Rate (CFAR) vessel detector on SAR ocean pixels.',
        'Cross-correlated coordinates and derived multi-sensor intelligence summary.',
      ],
      evidence: [
        {
          id: 'ev-sar-1',
          label: 'SAR Cloud Penetration',
          confidence: 0.97,
          category: 'radar_backscatter',
          description: '5.4 GHz C-band radar wavelength is unaffected by atmospheric hydrometeors and cloud vapor.',
        },
        {
          id: 'ev-sar-2',
          label: 'Point Scatterer Vessel Detection',
          confidence: 0.92,
          category: 'radar_backscatter',
          description: 'Bright specular returns (> -5 dB) pinpointing 4 steel-hulled cargo or fishing vessels.',
        },
        {
          id: 'ev-sar-3',
          label: 'Clean Shoreline Demarcation',
          confidence: 0.94,
          category: 'water',
          description: 'Sharp dielectric contrast between saline water (absorptive/specular) and rough terrain.',
        },
      ],
      coordinates: {
        lat: 18.920,
        lng: 72.780,
        zoom: 11,
        locationName: 'Arabian Sea Maritime Channel & Approaches, India',
        crs: 'EPSG:4326',
      },
      boundingBox: {
        north: 19.12,
        south: 18.72,
        east: 72.98,
        west: 72.58,
      },
      sensorMetadata: {
        platform: 'Sentinel-1A (C-SAR) + Sentinel-2A (MSI)',
        sensor: 'C-Band Active Synthetic Aperture Radar (VV+VH)',
        gsd: '10m / pixel',
        crs: 'EPSG:4326 (WGS 84)',
        acquisitionDate: '2026-07-21 UTC (Peak Monsoon Cloud Cover)',
        bands: ['C-Band 5.405 GHz (Radar)', 'MSI RGB Bands 4-3-2'],
      },
      geojson: [
        {
          type: 'Feature',
          properties: { name: 'Vessel Target 1 (Cargo Carrier)', class: 'radar_target', color: '#38bdf8' },
          geometry: { type: 'Point', coordinates: [72.75, 18.95] }
        },
        {
          type: 'Feature',
          properties: { name: 'Vessel Target 2 (Tanker)', class: 'radar_target', color: '#38bdf8' },
          geometry: { type: 'Point', coordinates: [72.82, 18.89] }
        },
        {
          type: 'Feature',
          properties: { name: 'Vessel Target 3 (Fishing Trawler)', class: 'radar_target', color: '#38bdf8' },
          geometry: { type: 'Point', coordinates: [72.71, 18.84] }
        },
        {
          type: 'Feature',
          properties: { name: 'Vessel Target 4 (Patrol Boat)', class: 'radar_target', color: '#38bdf8' },
          geometry: { type: 'Point', coordinates: [72.88, 18.81] }
        }
      ],
    },
  },
];
