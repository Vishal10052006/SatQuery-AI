import React from 'react';
import { Satellite, Server, ShieldCheck, Sparkles, Activity, Radio, Globe2 } from 'lucide-react';

interface NavbarProps {
  isDemoMode: boolean;
  setIsDemoMode: (val: boolean) => void;
  backendOnline: boolean;
  onOpenInfoModal: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  isDemoMode,
  setIsDemoMode,
  backendOnline,
  onOpenInfoModal,
}) => {
  return (
    <>
      {/* Top Orbital Telemetry Ticker */}
      <div className="telemetry-ticker">
        <div className="telemetry-item">
          <Activity size={12} color="var(--cyan)" />
          <span>MISSION: <span className="telemetry-val">SIH26167</span></span>
        </div>
        <div className="telemetry-item">
          <Globe2 size={12} color="var(--emerald)" />
          <span>ORBIT: <span className="telemetry-val">SENTINEL-2A (786 KM SSO)</span></span>
        </div>
        <div className="telemetry-item">
          <Radio size={12} color="var(--indigo)" />
          <span>RADAR: <span className="telemetry-val">SENTINEL-1 C-BAND SAR ACTIVE</span></span>
        </div>
        <div className="telemetry-item">
          <span>GSD: <span className="telemetry-val">10M/PX MULTISPECTRAL</span></span>
        </div>
        <div className="telemetry-item">
          <span>CRS: <span className="telemetry-val">EPSG:4326 (WGS 84)</span></span>
        </div>
        <div className="telemetry-item">
          <span>MODULE: <span style={{ color: 'var(--emerald-bright)' }}>M6 FRONTEND / GIS</span></span>
        </div>
      </div>

      {/* Main Glass Header */}
      <header className="navbar">
        <div className="navbar-brand">
          <div className="brand-icon-wrapper">
            <Satellite size={22} />
          </div>
          <div>
            <div className="brand-title">
              <span>SatQuery AI</span>
              <span className="brand-badge">SIH26167</span>
            </div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
              Geospatial Multi-Modal Intelligence • Earth Observation Command Center
            </div>
          </div>
        </div>

        <div className="navbar-actions">
          {/* Live Backend vs Demo Mode Toggle */}
          <div className="mode-toggle-pill">
            <button
              type="button"
              className={`mode-toggle-btn ${isDemoMode ? 'active' : ''}`}
              onClick={() => setIsDemoMode(true)}
              title="Use predefined sample satellite scenarios without requiring M5 FastAPI backend"
            >
              <Sparkles size={13} />
              Demo Mode
            </button>
            <button
              type="button"
              className={`mode-toggle-btn ${!isDemoMode ? 'active' : ''}`}
              onClick={() => setIsDemoMode(false)}
              title="Connect directly to M5 FastAPI backend at http://localhost:8000"
            >
              <Server size={13} />
              Live Backend
            </button>
          </div>

          {/* Backend Connectivity Status Indicator */}
          <div className="status-indicator">
            <div
              className={`status-dot ${
                isDemoMode ? 'demo' : backendOnline ? 'online' : 'offline'
              }`}
            />
            <span style={{ fontSize: '0.78rem' }}>
              {isDemoMode
                ? 'Demo Mode (Mock Data)'
                : backendOnline
                ? 'M5 Live: Connected'
                : 'M5 Live: Disconnected'}
            </span>
          </div>

          {/* SIH Team Info Button */}
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={onOpenInfoModal}
            title="View SIH Team Project Architecture & Responsibilities"
          >
            <ShieldCheck size={14} color="var(--cyan)" />
            Team Architecture
          </button>
        </div>
      </header>
    </>
  );
};
