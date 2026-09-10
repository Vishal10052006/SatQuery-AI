import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { Sidebar } from './components/Sidebar';
import { DashboardOverview } from './components/DashboardOverview';
import { ImageUploader } from './components/ImageUploader';
import { QueryConsole } from './components/QueryConsole';
import { StatusBar } from './components/StatusBar';
import { SingleImageViewer } from './components/SingleImageViewer';
import { ComparisonViewer } from './components/ComparisonViewer';
import { OpticalSarViewer } from './components/OpticalSarViewer';
import { EvidenceConfidencePanel } from './components/EvidenceConfidencePanel';
import { HistoryView } from './components/HistoryView';
import { TeamModal } from './components/TeamModal';
import { ReportModal } from './components/ReportModal';
import type { ViewMode, PipelineStage, AnalysisResponse } from './types';
import { SAMPLE_SCENARIOS } from './services/mockData';
import { executeSatelliteAnalysis, checkBackendHealth } from './services/api';

interface SlotState {
  file?: File;
  previewUrl?: string;
  name: string;
}

export const App: React.FC = () => {
  // Navigation and Modal State
  const [currentMode, setCurrentMode] = useState<ViewMode>('dashboard');
  const [isDemoMode, setIsDemoMode] = useState<boolean>(true);
  const [backendOnline, setBackendOnline] = useState<boolean>(false);
  const [showTeamModal, setShowTeamModal] = useState<boolean>(false);
  const [showReportModal, setShowReportModal] = useState<boolean>(false);

  // Active Inputs
  const [query, setQuery] = useState<string>('');
  const [slot1, setSlot1] = useState<SlotState>({ name: '' });
  const [slot2, setSlot2] = useState<SlotState>({ name: '' });

  // Execution & Output State
  const [pipelineStage, setPipelineStage] = useState<PipelineStage>('idle');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [currentResult, setCurrentResult] = useState<AnalysisResponse | null>(null);
  const [history, setHistory] = useState<AnalysisResponse[]>([]);

  // Check backend health periodically or on mode toggle
  useEffect(() => {
    const runHealthCheck = async () => {
      const status = await checkBackendHealth();
      setBackendOnline(status.isOnline);
    };
    runHealthCheck();
    const interval = setInterval(runHealthCheck, 15000);
    return () => clearInterval(interval);
  }, []);

  // Pre-load default demo scenario when switching modes if slots are empty
  const handleSelectMode = (newMode: ViewMode) => {
    setCurrentMode(newMode);
    setErrorMessage(null);

    // If navigating to a specific analysis mode and inputs are empty, load the recommended scenario
    if (newMode !== 'dashboard' && newMode !== 'history') {
      const scenario = SAMPLE_SCENARIOS.find((s) => s.mode === newMode);
      if (scenario && !slot1.previewUrl) {
        loadScenario(scenario.id, false);
      }
    }
  };

  // Launch scenario from Dashboard or quick actions
  const loadScenario = (scenarioId: string, switchView = true) => {
    const scenario = SAMPLE_SCENARIOS.find((s) => s.id === scenarioId);
    if (!scenario) return;

    if (switchView) {
      setCurrentMode(scenario.mode);
    }
    setQuery(scenario.query);
    setErrorMessage(null);

    if (scenario.images.primary) {
      setSlot1({ previewUrl: scenario.images.primary, name: 'Sample Sentinel-2 MSI RGB' });
      setSlot2({ name: '' });
    } else if (scenario.images.before && scenario.images.after) {
      setSlot1({ previewUrl: scenario.images.before, name: 'Pre-Event Baseline (T1)' });
      setSlot2({ previewUrl: scenario.images.after, name: 'Post-Event Inundation (T2)' });
    } else if (scenario.images.optical && scenario.images.sar) {
      setSlot1({ previewUrl: scenario.images.optical, name: 'Optical Sentinel-2 RGB' });
      setSlot2({ previewUrl: scenario.images.sar, name: 'SAR Sentinel-1 C-Band VV+VH' });
    }

    // Set sample result right away for instant preview
    setCurrentResult(scenario.mockResponse);
  };

  // Trigger analysis pipeline
  const handleAnalyze = async () => {
    setErrorMessage(null);

    // Validation
    if (!query.trim()) {
      setErrorMessage('Please enter a natural language query before running analysis.');
      return;
    }

    if (currentMode === 'image-understanding' && !slot1.previewUrl) {
      setErrorMessage('Please upload a satellite image or click "Load Land-Use Satellite Scene".');
      return;
    }

    if (
      (currentMode === 'change-detection' || currentMode === 'optical-sar') &&
      (!slot1.previewUrl || !slot2.previewUrl)
    ) {
      setErrorMessage('This mode requires two images. Please provide both image inputs or load the demo pair.');
      return;
    }

    try {
      // Step 1: Routing (M4)
      setPipelineStage('routing');
      await new Promise((res) => setTimeout(res, 450));

      // Step 2: Model Inference (M1/M2/M3)
      setPipelineStage('analyzing');
      await new Promise((res) => setTimeout(res, 550));

      // Step 3: Evidence & Confidence synthesis
      setPipelineStage('evidence');

      const response = await executeSatelliteAnalysis(
        {
          mode: currentMode,
          query: query.trim(),
          primaryImage: slot1,
          beforeImage: slot1,
          afterImage: slot2,
          opticalImage: slot1,
          sarImage: slot2,
        },
        isDemoMode
      );

      setCurrentResult(response);
      setHistory((prev) => [response, ...prev]);
      setPipelineStage('completed');
    } catch (err: any) {
      setPipelineStage('error');
      setErrorMessage(
        err.message || 'Analysis pipeline execution failed. Please verify API configuration or enable Demo Mode.'
      );
    }
  };

  const isAnalysisDisabled =
    !query.trim() ||
    (currentMode === 'image-understanding' && !slot1.previewUrl) ||
    ((currentMode === 'change-detection' || currentMode === 'optical-sar') &&
      (!slot1.previewUrl || !slot2.previewUrl));

  return (
    <div className="app-layout">
      <Navbar
        isDemoMode={isDemoMode}
        setIsDemoMode={setIsDemoMode}
        backendOnline={backendOnline}
        onOpenInfoModal={() => setShowTeamModal(true)}
      />

      <div className="main-body">
        <Sidebar
          currentMode={currentMode}
          onSelectMode={handleSelectMode}
          historyCount={history.length}
        />

        <main className="content-container">
          {/* Status & Alerts */}
          <StatusBar
            stage={pipelineStage}
            error={errorMessage}
            onDismissError={() => setErrorMessage(null)}
            onSwitchToDemo={() => {
              setIsDemoMode(true);
              setErrorMessage(null);
            }}
            isDemoMode={isDemoMode}
          />

          {/* View 1: Main Dashboard Overview */}
          {currentMode === 'dashboard' && (
            <DashboardOverview
              onSelectMode={handleSelectMode}
              onLaunchDemoScenario={loadScenario}
              currentResult={currentResult}
              pipelineStage={pipelineStage}
              history={history}
              onOpenReport={() => setShowReportModal(true)}
            />
          )}

          {/* View 2: Image Understanding (Single VLM) */}
          {currentMode === 'image-understanding' && (
            <div>
              <div style={{ marginBottom: 18 }}>
                <h2 style={{ fontSize: '1.25rem', fontWeight: 600 }}>Image Understanding & Semantic VLM</h2>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  Visual Question Answering (VQA) for single-date satellite scenes. Detect land-use, infrastructure, and natural features.
                </p>
              </div>

              <ImageUploader
                mode="image-understanding"
                slot1={slot1}
                slot2={slot2}
                onChangeSlot1={setSlot1}
                onChangeSlot2={setSlot2}
                onLoadSample={() => loadScenario('land-use-analysis', false)}
                sampleLabel="Load Land-Use Satellite Scene"
              />

              <QueryConsole
                query={query}
                setQuery={setQuery}
                mode="image-understanding"
                onAnalyze={handleAnalyze}
                isLoading={pipelineStage !== 'idle' && pipelineStage !== 'completed' && pipelineStage !== 'error'}
                disabled={isAnalysisDisabled}
              />

              {currentResult && currentResult.mode === 'image-understanding' && (
                <div>
                  <SingleImageViewer
                    imageUrl={currentResult.primaryImageUrl || slot1.previewUrl || ''}
                    overlayUrl={currentResult.overlayImageUrl}
                    title="Sentinel-2 Multispectral Scene"
                    evidence={currentResult.evidence}
                    coordinates={currentResult.coordinates}
                    boundingBox={currentResult.boundingBox}
                    geojson={currentResult.geojson}
                    sensorMetadata={currentResult.sensorMetadata}
                  />
                  <EvidenceConfidencePanel
                    response={currentResult}
                    onOpenReport={() => setShowReportModal(true)}
                  />
                </div>
              )}
            </div>
          )}

          {/* View 3: Change Detection (Before / After) */}
          {currentMode === 'change-detection' && (
            <div>
              <div style={{ marginBottom: 18 }}>
                <h2 style={{ fontSize: '1.25rem', fontWeight: 600 }}>Bi-Temporal Change Detection</h2>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  Pair pre-event and post-event imagery to evaluate environmental impacts, flooding, and spatial structural alterations.
                </p>
              </div>

              <ImageUploader
                mode="change-detection"
                slot1={slot1}
                slot2={slot2}
                onChangeSlot1={setSlot1}
                onChangeSlot2={setSlot2}
                onLoadSample={() => loadScenario('change-detection-flood', false)}
                sampleLabel="Load Flood Before/After Pair"
              />

              <QueryConsole
                query={query}
                setQuery={setQuery}
                mode="change-detection"
                onAnalyze={handleAnalyze}
                isLoading={pipelineStage !== 'idle' && pipelineStage !== 'completed' && pipelineStage !== 'error'}
                disabled={isAnalysisDisabled}
              />

              {currentResult && currentResult.mode === 'change-detection' && (
                <div>
                  <ComparisonViewer
                    beforeUrl={currentResult.beforeImageUrl || slot1.previewUrl || ''}
                    afterUrl={currentResult.afterImageUrl || slot2.previewUrl || ''}
                    overlayUrl={currentResult.overlayImageUrl}
                    changes={currentResult.changes}
                    beforeTitle="Timestamp T1: Pre-Event Baseline"
                    afterTitle="Timestamp T2: Post-Event Inundation"
                    coordinates={currentResult.coordinates}
                    boundingBox={currentResult.boundingBox}
                    geojson={currentResult.geojson}
                    sensorMetadata={currentResult.sensorMetadata}
                  />
                  <EvidenceConfidencePanel
                    response={currentResult}
                    onOpenReport={() => setShowReportModal(true)}
                  />
                </div>
              )}
            </div>
          )}

          {/* View 4: Optical + SAR Comparison */}
          {currentMode === 'optical-sar' && (
            <div>
              <div style={{ marginBottom: 18 }}>
                <h2 style={{ fontSize: '1.25rem', fontWeight: 600 }}>Optical + SAR Multi-Sensor Fusion</h2>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  Overcome cloud obstruction and fog by synchronizing Sentinel-2 Optical reflectance with Sentinel-1 Synthetic Aperture Radar.
                </p>
              </div>

              <ImageUploader
                mode="optical-sar"
                slot1={slot1}
                slot2={slot2}
                onChangeSlot1={setSlot1}
                onChangeSlot2={setSlot2}
                onLoadSample={() => loadScenario('optical-sar-comparison', false)}
                sampleLabel="Load Optical & SAR Scene Pair"
              />

              <QueryConsole
                query={query}
                setQuery={setQuery}
                mode="optical-sar"
                onAnalyze={handleAnalyze}
                isLoading={pipelineStage !== 'idle' && pipelineStage !== 'completed' && pipelineStage !== 'error'}
                disabled={isAnalysisDisabled}
              />

              {currentResult && currentResult.mode === 'optical-sar' && (
                <div>
                  <OpticalSarViewer
                    opticalUrl={currentResult.opticalImageUrl || slot1.previewUrl || ''}
                    sarUrl={currentResult.sarImageUrl || slot2.previewUrl || ''}
                    opticalTitle="Sentinel-2 Optical (Cloud Obscured)"
                    sarTitle="Sentinel-1 SAR C-Band (All-Weather Penetration)"
                    coordinates={currentResult.coordinates}
                    boundingBox={currentResult.boundingBox}
                    geojson={currentResult.geojson}
                    sensorMetadata={currentResult.sensorMetadata}
                  />
                  <EvidenceConfidencePanel
                    response={currentResult}
                    onOpenReport={() => setShowReportModal(true)}
                  />
                </div>
              )}
            </div>
          )}

          {/* View 5: Analysis History */}
          {currentMode === 'history' && (
            <HistoryView
              history={history}
              onSelectHistoryItem={(item) => {
                setCurrentResult(item);
                setCurrentMode(item.mode);
                setQuery(item.query);
                if (item.primaryImageUrl) setSlot1({ previewUrl: item.primaryImageUrl, name: 'Loaded from History' });
                if (item.beforeImageUrl) setSlot1({ previewUrl: item.beforeImageUrl, name: 'Before (History)' });
                if (item.afterImageUrl) setSlot2({ previewUrl: item.afterImageUrl, name: 'After (History)' });
                if (item.opticalImageUrl) setSlot1({ previewUrl: item.opticalImageUrl, name: 'Optical (History)' });
                if (item.sarImageUrl) setSlot2({ previewUrl: item.sarImageUrl, name: 'SAR (History)' });
              }}
              onClearHistory={() => setHistory([])}
            />
          )}
        </main>
      </div>

      {/* Team Architecture Modal */}
      <TeamModal
        isOpen={showTeamModal}
        onClose={() => setShowTeamModal(false)}
      />

      {/* Official SIH Report Modal */}
      <ReportModal
        isOpen={showReportModal}
        onClose={() => setShowReportModal(false)}
        response={currentResult}
      />
    </div>
  );
};

export default App;
