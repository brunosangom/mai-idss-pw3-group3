import React from 'react';
import { Link } from 'react-router-dom';
import '../components/Content.css';
import Card from '../components/Card';

function NotFound() {
  return (
    <div className="content-grid">
      <Card title="Page not found">
        <p>We couldn't find that page.</p>
        <p>
          <Link to="/" className="sidebar-link">Go back to Overview</Link>
        </p>
      </Card>
    </div>
  );
}

export default NotFound;
