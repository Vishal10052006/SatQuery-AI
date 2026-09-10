import { AlertTriangle, Loader2, Sparkles } from 'lucide-react';
import type { PipelineStage } from '../types';

interface StatusBarProps {
  stage: PipelineStage;
  error: string | null;
  onDismissError: () => void;
  onSwitchToDemo: () => void;
  isDemoMode: boolean;
}

export const StatusBar: React.FC<StatusBarProps> = ({
  stage,
  error,
  onDismissError,
  onSwitchToDemo,
  isDemoMode,
}) => {
  if (error) {
    return (
      <div className="alert alert-danger" style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', gap: 12 }}>
          <AlertTriangle size={20} color="var(--rose)" style={{ flexShrink: 0, marginTop: 2 }} />
          <div>
            <div style={{ fontWeight: 600, marginBottom: 2 }}>Analysis Pipeline Error</div>
            <div style={{ fontSize: '0.85rem' }}>{error}</div>
            {!isDemoMode && (
              <div style={{ marginTop: 8, fontSize: '0.8rem' }}>
                Note: Since M5's live FastAPI backend is currently unavailable, switch to <strong>Demo Mode</strong> to test with realistic mock satellite data.
              </div>
            )}
          </div>
        </div>

        <div style={{ display: 'flex', gap: 8, flexShrink: 0 }}>
          {!isDemoMode && (
            <button
              type="button"
              className="btn btn-sm"
              onClick={onSwitchToDemo}
              style={{ background: 'var(--cyan)', color: '#041019', fontWeight: 600 }}
            >
              <Sparkles size={13} /> Switch to Demo Mode
            </button>
          )}
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={onDismissError}
          >
            Dismiss
          </button>
        </div>
      </div>
    );
  }

  if (stage === 'idle' || stage === 'completed') {
    return null;
  }

  // Active Loading Pipeline Stepper
  return (
    <div className="progress-stepper">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.88rem', fontWeight: 600 }}>
          <Loader2 size={16} className="animate-spin" color="var(--cyan)" />
          SatQuery AI Inference Pipeline Active
        </div>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          Coordinating multi-modal inference
        </div>
      </div>

      <div className="stepper-row">
        <div className={`stepper-step ${stage === 'routing' ? 'active' : 'completed'}`}>
          <div className="step-bubble">1</div>
          <div>
            <div>Query Classification</div>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>M4 Agentic Routing</div>
          </div>
        </div>

        <div style={{ color: 'var(--border-medium)' }}>➔</div>

        <div className={`stepper-step ${stage === 'analyzing' ? 'active' : stage === 'evidence' ? 'completed' : ''}`}>
          <div className="step-bubble">2</div>
          <div>
            <div>Multi-Modal Model</div>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>M1/M2/M3 Inference</div>
          </div>
        </div>

        <div style={{ color: 'var(--border-medium)' }}>➔</div>

        <div className={`stepper-step ${stage === 'evidence' ? 'active' : ''}`}>
          <div className="step-bubble">3</div>
          <div>
            <div>Evidence & Overlays</div>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Confidence Synthesis</div>
          </div>
        </div>
      </div>
    </div>
  );
};
