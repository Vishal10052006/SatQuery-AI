import { Send, Terminal, Sparkles, Loader2, RotateCcw } from 'lucide-react';
import type { ViewMode } from '../types';

interface QueryConsoleProps {
  query: string;
  setQuery: (val: string) => void;
  mode: ViewMode;
  onAnalyze: () => void;
  isLoading: boolean;
  disabled: boolean;
}

export const QueryConsole: React.FC<QueryConsoleProps> = ({
  query,
  setQuery,
  mode,
  onAnalyze,
  isLoading,
  disabled,
}) => {
  const getPromptSuggestions = () => {
    switch (mode) {
      case 'image-understanding':
        return [
          'Identify the major land-use features in this image.',
          'Delineate the coastal water boundary and urban clusters.',
          'Quantify agricultural acreage versus built-up infrastructure.',
        ];
      case 'change-detection':
        return [
          'What changed between these two images?',
          'Highlight river overbank flood extent and submerged acreage.',
          'Detect infrastructure damage and road washouts between T1 and T2.',
        ];
      case 'optical-sar':
        return [
          'Compare the optical and SAR imagery.',
          'Detect maritime vessels obscured under dense cloud cover.',
          'Explain why the water surface appears dark in the SAR backscatter.',
        ];
      default:
        return [
          'Identify the major land-use features in this image.',
          'What changed between these two images?',
          'Compare the optical and SAR imagery.',
        ];
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      if (!disabled && !isLoading) {
        onAnalyze();
      }
    }
  };

  return (
    <div className="query-box">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Terminal size={18} color="var(--cyan)" />
          <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>Natural Language Geospatial Query</span>
        </div>
        {query && (
          <button
            type="button"
            className="btn btn-sm"
            onClick={() => setQuery('')}
            style={{ color: 'var(--text-muted)', background: 'transparent', padding: '2px 8px' }}
            title="Clear prompt"
          >
            <RotateCcw size={12} /> Clear Query
          </button>
        )}
      </div>

      <textarea
        className="query-textarea"
        placeholder="Enter your satellite reasoning question (e.g., 'Identify the major land-use features in this image.')..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={isLoading}
      />

      {/* Suggested Prompt Chips */}
      <div style={{ marginTop: 12 }}>
        <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 6, display: 'flex', alignItems: 'center', gap: 4 }}>
          <Sparkles size={12} color="var(--cyan)" /> Recommended Prompt Scenarios:
        </div>
        <div className="prompt-chips-row">
          {getPromptSuggestions().map((prompt, idx) => (
            <button
              key={idx}
              type="button"
              className="prompt-chip"
              onClick={() => setQuery(prompt)}
              disabled={isLoading}
            >
              "{prompt}"
            </button>
          ))}
        </div>
      </div>

      {/* Submit Button Bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 16, paddingTop: 12, borderTop: '1px solid var(--border-subtle)' }}>
        <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
          Tip: Press <kbd style={{ background: 'var(--bg-card-elevated)', padding: '2px 5px', borderRadius: 4, border: '1px solid var(--border-medium)', fontFamily: 'var(--font-mono)' }}>Ctrl+Enter</kbd> to analyze
        </span>

        <button
          type="button"
          className="btn btn-primary"
          onClick={onAnalyze}
          disabled={disabled || isLoading}
          style={{ minWidth: 190 }}
        >
          {isLoading ? (
            <>
              <Loader2 size={16} className="animate-spin" />
              Running SatQuery AI...
            </>
          ) : (
            <>
              <Send size={15} />
              Analyze Satellite Imagery
            </>
          )}
        </button>
      </div>
    </div>
  );
};
