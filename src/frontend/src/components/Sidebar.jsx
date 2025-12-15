import React from 'react';

import SidebarItem from './SidebarItem';
import { FiHome, FiBarChart2, FiTrendingUp, FiClock, FiChevronLeft, FiChevronRight } from 'react-icons/fi';
import './Sidebar.css';

function Sidebar({ collapsed = false, onToggle }) {
  return (
    <div className={`sidebar ${collapsed ? 'is-collapsed' : ''}`}>
      <div className="sidebar-header">
        <h2>Wildfire Dashboard</h2>
        <button
          type="button"
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          className="sidebar-toggle"
          onClick={onToggle}
        >
          {collapsed ? <FiChevronRight className="sidebar-icon" /> : <FiChevronLeft className="sidebar-icon" />}
        </button>
      </div>
      <div className="sidebar-divider" aria-hidden="true" />
      <nav>
        <ul>
          <SidebarItem to="/" end icon={FiHome} label="Overview" />
          <SidebarItem to="/statistics" icon={FiBarChart2} label="Statistics" />
          <SidebarItem to="/forecast" icon={FiTrendingUp} label="Forecast" />
          <SidebarItem to="/historical" icon={FiClock} label="Historical Data" />
        </ul>
      </nav>
    </div>
  );
}

export default Sidebar;