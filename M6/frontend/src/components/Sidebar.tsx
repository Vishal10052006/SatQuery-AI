import React from 'react';
import { 
  LayoutDashboard, 
  Eye, 
  SplitSquareVertical, 
  Layers, 
  History, 
  Cpu
} from 'lucide-react';
import type { ViewMode } from '../types';

interface SidebarProps {
  currentMode: ViewMode;
  onSelectMode: (mode: ViewMode) => void;
  historyCount: number;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentMode,
  onSelectMode,
  historyCount,
}) => {
  const menuItems = [
    {
      id: 'dashboard' as ViewMode,
      label: 'Main Dashboard',
      icon: <LayoutDashboard size={18} />,
      badge: 'Overview',
    },
    {
      id: 'image-understanding' as ViewMode,
      label: 'Image Understanding',
      icon: <Eye size={18} />,
      badge: 'Single VLM',
    },
    {
      id: 'change-detection' as ViewMode,
      label: 'Change Detection',
      icon: <SplitSquareVertical size={18} />,
      badge: 'Before / After',
    },
    {
      id: 'optical-sar' as ViewMode,
      label: 'Optical + SAR',
      icon: <Layers size={18} />,
      badge: 'Multi-Sensor',
    },
    {
      id: 'history' as ViewMode,
      label: 'Analysis History',
      icon: <History size={18} />,
      badge: historyCount > 0 ? `${historyCount}` : undefined,
    },
  ];

  return (
    <aside className="sidebar">
      <ul className="sidebar-menu">
        {menuItems.map((item) => {
          const isActive = currentMode === item.id;
          return (
            <li key={item.id}>
              <button
                type="button"
                className={`sidebar-item ${isActive ? 'active' : ''}`}
                onClick={() => onSelectMode(item.id)}
                style={{ width: '100%', textAlign: 'left', background: 'transparent' }}
              >
                {item.icon}
                <span style={{ flex: 1 }}>{item.label}</span>
                {item.badge && (
                  <span
                    style={{
                      fontSize: '0.68rem',
                      padding: '2px 6px',
                      borderRadius: 'var(--radius-full)',
                      background: isActive ? 'var(--cyan)' : 'var(--bg-card-elevated)',
                      color: isActive ? '#041019' : 'var(--text-muted)',
                      fontWeight: 600,
                    }}
                  >
                    {item.badge}
                  </span>
                )}
              </button>
            </li>
          );
        })}
      </ul>

      <div className="sidebar-footer">
        <div className="team-badge-box">
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
            <Cpu size={14} color="var(--cyan)" />
            <span className="team-badge-title">SIH Team Architecture</span>
          </div>
          <div className="team-badge-role">
            Active Module: <strong>M6 (Frontend / GIS)</strong>
          </div>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 4 }}>
            Connected to M1-M5 Pipeline
          </div>
        </div>
      </div>
    </aside>
  );
};
