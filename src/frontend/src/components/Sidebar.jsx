import React from 'react';
import { NavLink } from 'react-router-dom';
import './Sidebar.css';

function Sidebar() {
  return (
    <div className="sidebar">
      <h2>Wildfire Dashboard</h2>
      <nav>
        <ul>
          <li>
            <NavLink
              to="/"
              end
              className={({ isActive }) => isActive ? 'sidebar-link sidebar-link--active' : 'sidebar-link'}
            >
              Overview
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/statistics"
              className={({ isActive }) => isActive ? 'sidebar-link sidebar-link--active' : 'sidebar-link'}
            >
              Statistics
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/forecast"
              className={({ isActive }) => isActive ? 'sidebar-link sidebar-link--active' : 'sidebar-link'}
            >
              Forecast
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/historical"
              className={({ isActive }) => isActive ? 'sidebar-link sidebar-link--active' : 'sidebar-link'}
            >
              Historical Data
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/map"
              className={({ isActive }) => isActive ? 'sidebar-link sidebar-link--active' : 'sidebar-link'}
            >
              Map
            </NavLink>
          </li>
        </ul>
      </nav>
    </div>
  );
}

export default Sidebar;