import { useState } from 'react';
import { 
  ShieldCheck, 
  Sparkles, 
  Copy, 
  Check, 
  ChevronDown, 
  ChevronUp, 
  Cpu, 
  FileText,
  Activity
} from 'lucide-react';
import type { AnalysisResponse, EvidenceItem } from '../types';

interface EvidenceConfidencePanelProps {
  response: AnalysisResponse;
  onOpenReport?: () => void;
}

export const EvidenceConfidencePanel: React.FC<EvidenceConfidencePanelProps> = ({
  response,
  onOpenReport,
}) => {
  const [copied, setCopied] = useState(false);
  const [showReasoning, setShowReasoning] = useState(false);

  const handleCopyAnswer = () => {
    navigator.clipboard.writeText(response.answer);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const confidencePercent = response.confidence != null 
    ? Math.round(response.confidence * 100) 
    : null;

  const getConfidenceRating = (percent: number) => {
    if (percent >= 90) return { label: 'High Confidence', color: 'var(--emerald)' };
    if (percent >= 75) return { label: 'Moderate Confidence', color: 'var(--cyan)' };
    return { label: 'Low Confidence / Ambiguous', color: 'var(--amber)' };
  };

  const rating = confidencePercent != null ? getConfidenceRating(confidencePercent) : null;

  const getCategoryColor = (cat: EvidenceItem['category']) => {
    switch (cat) {
      case 'urban': return 'var(--rose)';
      case 'water': return 'var(--cyan)';
      case 'vegetation': return 'var(--emerald)';
      case 'infrastructure': return 'var(--amber)';
      case 'radar_backscatter': return 'var(--indigo)';
      case 'change': return '#e11d48';
      default: return 'var(--text-secondary)';
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Primary Answer Card */}
      <div className="card">
        <div className="card-header">
          <div>
            <div className="card-title">
              <Sparkles size={18} color="var(--cyan)" />
              SatQuery AI Intelligence Summary
            </div>
            <div className="card-subtitle">
              User Query: <span style={{ color: 'var(--text-primary)', fontStyle: 'italic' }}>"{response.query}"</span>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {response.modelUsed && (
              <span className="badge badge-cyan" style={{ fontSize: '0.72rem' }}>
                <Cpu size={12} /> {response.modelUsed}
              </span>
            )}
            {onOpenReport && (
              <button
                type="button"
                className="btn btn-primary btn-sm"
                onClick={onOpenReport}
                title="Generate Official Mission Report"
              >
                <FileText size={13} />
                Generate SIH Report
              </button>
            )}
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={handleCopyAnswer}
              title="Copy answer text"
            >
              {copied ? <Check size={13} color="var(--emerald)" /> : <Copy size={13} />}
              {copied ? 'Copied' : 'Copy'}
            </button>
          </div>
        </div>

        {/* Answer Content */}
        <div style={{ fontSize: '0.94rem', lineHeight: 1.65, color: '#f1f5f9', background: 'var(--bg-input)', padding: '16px 20px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)', marginBottom: 14 }}>
          {response.answer}
        </div>

        {/* Summary Pill */}
        {response.resultSummary && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.82rem', color: 'var(--cyan)', background: 'var(--cyan-glow)', padding: '8px 14px', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(6, 182, 212, 0.25)' }}>
            <Activity size={15} />
            <span><strong>Executive Result:</strong> {response.resultSummary}</span>
          </div>
        )}
      </div>

      {/* Confidence & Evidence Card */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <ShieldCheck size={18} color="var(--emerald)" />
            Evidence & Confidence Verification
          </div>
          {response.executionTimeMs && (
            <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              Latency: {response.executionTimeMs}ms
            </span>
          )}
        </div>

        {/* Confidence Gauge */}
        {confidencePercent != null ? (
          <div className="confidence-meter-container">
            <div className="confidence-circle" style={{ borderColor: rating?.color, color: rating?.color }}>
              <span>{confidencePercent}%</span>
            </div>
            <div className="confidence-info">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span className="confidence-title">Model Confidence Score</span>
                <span style={{ fontSize: '0.78rem', fontWeight: 600, color: rating?.color }}>
                  {rating?.label}
                </span>
              </div>
              <div className="confidence-bar-bg">
                <div 
                  className="confidence-bar-fill" 
                  style={{ width: `${confidencePercent}%`, background: rating?.color }} 
                />
              </div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 4 }}>
                Calculated across multi-modal attention weights and cross-sensor feature consistency.
              </div>
            </div>
          </div>
        ) : (
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontStyle: 'italic', marginBottom: 12 }}>
            Confidence score not provided in backend response.
          </div>
        )}

        {/* Extracted Spatial Evidence Items */}
        <div>
          <div style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
            <FileText size={15} color="var(--cyan)" />
            Identified Geospatial Evidence ({response.evidence?.length || 0} features)
          </div>

          {response.evidence && response.evidence.length > 0 ? (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 10 }}>
              {response.evidence.map((item) => (
                <div
                  key={item.id}
                  style={{
                    background: 'var(--bg-input)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: 'var(--radius-md)',
                    padding: '10px 14px',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
                    <span style={{ fontSize: '0.82rem', fontWeight: 600, color: getCategoryColor(item.category) }}>
                      {item.label}
                    </span>
                    {item.confidence != null && (
                      <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                        {Math.round(item.confidence * 100)}% conf
                      </span>
                    )}
                  </div>
                  <div style={{ fontSize: '0.76rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                    {item.description}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
              No explicit bounding evidence returned for this query.
            </div>
          )}
        </div>

        {/* Reasoning Pipeline Accordion */}
        {response.reasoningSteps && response.reasoningSteps.length > 0 && (
          <div style={{ marginTop: 16, paddingTop: 14, borderTop: '1px solid var(--border-subtle)' }}>
            <button
              type="button"
              onClick={() => setShowReasoning(!showReasoning)}
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                background: 'transparent',
                border: 'none',
                color: 'var(--text-secondary)',
                fontSize: '0.8rem',
                cursor: 'pointer',
                fontWeight: 500,
              }}
            >
              <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <Cpu size={14} color="var(--cyan)" />
                Detailed Model Pipeline Reasoning Trace ({response.reasoningSteps.length} steps)
              </span>
              {showReasoning ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            </button>

            {showReasoning && (
              <ul style={{ listStyle: 'none', marginTop: 10, display: 'flex', flexDirection: 'column', gap: 6 }}>
                {response.reasoningSteps.map((step, idx) => (
                  <li
                    key={idx}
                    style={{
                      fontSize: '0.78rem',
                      color: 'var(--text-secondary)',
                      background: 'var(--bg-input)',
                      padding: '6px 12px',
                      borderRadius: 4,
                      borderLeft: '3px solid var(--cyan)',
                    }}
                  >
                    <strong style={{ color: 'var(--text-primary)', marginRight: 6 }}>Step {idx + 1}:</strong>
                    {step}
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
