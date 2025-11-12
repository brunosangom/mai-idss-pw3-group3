import { useEffect, useRef, useState } from 'react';

// Hook to adapt axis Bottom config based on available width.
// Rotates labels and skips some when space is constrained.
export function useAdaptiveAxis(labels) {
  const ref = useRef(null);
  const [width, setWidth] = useState(null);

  useEffect(() => {
    if (!ref.current) return;
    const obs = new ResizeObserver(entries => {
      for (const entry of entries) {
        setWidth(entry.contentRect.width);
      }
    });
    obs.observe(ref.current);
    return () => obs.disconnect();
  }, []);

  let tickRotation = 0;
  let tickValues = labels;

  if (width !== null) {
    if (width < 240) {
      tickRotation = -45; // rotate for very narrow containers
    }
    // Skip labels progressively if extremely narrow
    if (width < 200) {
      tickValues = labels.filter((_, i) => i % 2 === 0); // every other
    }
    if (width < 150) {
      tickValues = labels.filter((_, i) => i % 3 === 0); // every third
    }
  }

  const axisBottom = {
    tickSize: 0,
    tickPadding: 10,
    tickRotation,
    tickValues,
  };

  return { ref, axisBottom };
}
