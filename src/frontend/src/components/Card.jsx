import React from 'react';

function Card({ title, children, className = '', actions = null }) {
  const classes = className ? `card ${className}` : 'card';
  return (
    <section className={classes}>
      {(title || actions) && (
        <>
          <div className="card-header">
            {title ? <h2 className="card-title">{title}</h2> : <span />}
            {actions ? <div className="card-actions">{actions}</div> : null}
          </div>
          {children && <div className="card-divider" />}
        </>
      )}
      {children && children}
    </section>
  );
}

export default Card;
