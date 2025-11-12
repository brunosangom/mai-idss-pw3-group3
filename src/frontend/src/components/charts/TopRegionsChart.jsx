import React from 'react';
import { ResponsiveLine } from '@nivo/line';
import { CHART_SERIES_COLORS, PALETTE } from './Palette';

// Mock top 5 regions weekly counts (each series is one region across 7 days)
const regionNames = ['West', 'South', 'East', 'Central', 'North'];
const mockData = regionNames.map((name, idx) => ({
  id: name,
  color: CHART_SERIES_COLORS[idx % CHART_SERIES_COLORS.length],
  data: [
    { x: 'Mon', y: Math.round(20 + Math.random() * 25) },
    { x: 'Tue', y: Math.round(20 + Math.random() * 25) },
    { x: 'Wed', y: Math.round(20 + Math.random() * 25) },
    { x: 'Thu', y: Math.round(20 + Math.random() * 25) },
    { x: 'Fri', y: Math.round(20 + Math.random() * 25) },
    { x: 'Sat', y: Math.round(20 + Math.random() * 25) },
    { x: 'Sun', y: Math.round(20 + Math.random() * 25) }
  ]
}));

function TopRegionsChart() {
  return (
    <div className="chart-slot">
      <ResponsiveLine
        data={mockData}
        margin={{ top: 20, right: 20, bottom: 35, left: 50 }}
        xScale={{ type: 'point' }}
        yScale={{ type: 'linear', min: 'auto', max: 'auto', stacked: false }}
        colors={d => d.color}
        lineWidth={2}
        enablePoints={false}
        enableGridX={false}
        gridYValues={5}
        axisTop={null}
        axisRight={null}
        axisBottom={{ tickSize: 0, tickPadding: 10 }}
        axisLeft={{ tickSize: 0, tickPadding: 6 }}
        useMesh={true}
        theme={{
          axis: {
            ticks: { text: { fill: PALETTE.grayDark, fontSize: 11 } },
            domain: { line: { stroke: PALETTE.grayLight } },
            legend: { text: { fill: PALETTE.grayDark } }
          },
          grid: { line: { stroke: PALETTE.grayLight, strokeWidth: 1 } },
          tooltip: { container: { background: '#fff', color: PALETTE.grayDark, fontSize: 12 } }
        }}
        curve="monotoneX"
      />
    </div>
  );
}

export default TopRegionsChart;
