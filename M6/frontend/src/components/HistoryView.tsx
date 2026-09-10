import { History, Trash2, ArrowUpRight, Clock } from 'lucide-react';
import type { AnalysisResponse } from '../types';

interface HistoryViewProps {
  history: AnalysisResponse[];
  onSelectHistoryItem: (item: AnalysisResponse) => void;
  onClearHistory: () => void;
}

export const HistoryView: React.FC<HistoryViewProps> = ({
  history,
  onSelectHistoryItem,
  onClearHistory,
}) => {
  if (history.length === 0) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: '48px 24px' }}>
        <div style={{ width: 48, height: 48, borderRadius: '50%', background: 'var(--bg-card-elevated)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 16px', color: 'var(--text-muted)' }}>
          <History size={24} />
        </div>
        <h3 style={{ fontSize: '1.05rem', fontWeight: 600, marginBottom: 6 }}>No Analysis Runs Recorded Yet</h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', maxWidth: 460, margin: '0 auto' }}>
          When you execute queries on satellite scenes, the full response along with confidence scores and reasoning steps will be logged here.
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header">
        <div>
          <div className="card-title">
            <History size={18} color="var(--cyan)" />
            Recent Satellite Analysis Runs ({history.length})
          </div>
          <div className="card-subtitle">
            Click any entry to inspect its visualizations, evidence, and model reasoning.
          </div>
        </div>

        <button
          type="button"
          className="btn btn-secondary btn-sm"
          onClick={onClearHistory}
          style={{ color: 'var(--rose)' }}
        >
          <Trash2 size={13} /> Clear History
        </button>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {history.map((item) => (
          <div
            key={item.id}
            onClick={() => onSelectHistoryItem(item)}
            style={{
              background: 'var(--bg-card-elevated)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--radius-md)',
              padding: '14px 18px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              transition: 'all 0.15s ease',
            }}
            onMouseEnter={(e) => (e.currentTarget.style.borderColor = 'var(--cyan)')}
            onMouseLeave={(e) => (e.currentTarget.style.borderColor = 'var(--border-subtle)')}
          >
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4, flex: 1, marginRight: 16 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span
                  className="badge badge-cyan"
                  style={{ textTransform: 'capitalize', fontSize: '0.7rem' }}
                >
                  {item.mode.replace('-', ' ')}
                </span>
                <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 4 }}>
                  <Clock size={12} /> {new Date(item.timestamp).toLocaleTimeString()}
                </span>
              </div>
              <div style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                "{item.query}"
              </div>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 640 }}>
                {item.answer}
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
              {item.confidence != null && (
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--emerald)' }}>
                    {Math.round(item.confidence * 100)}%
                  </div>
                  <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>Confidence</div>
                </div>
              )}
              <ArrowUpRight size={18} color="var(--cyan)" />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
