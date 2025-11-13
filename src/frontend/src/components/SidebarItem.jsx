import React from 'react';
import { NavLink } from 'react-router-dom';

function SidebarItem({ to, label, icon: Icon, end = false }) {
  return (
    <li>
      <NavLink
        to={to}
        end={end}
        className={({ isActive }) =>
          isActive ? 'sidebar-link sidebar-link--active' : 'sidebar-link'
        }
      >
        {Icon ? <Icon className="sidebar-icon" aria-hidden="true" /> : null}
        <span className="sidebar-label">{label}</span>
      </NavLink>
    </li>
  );
}

export default SidebarItem;
