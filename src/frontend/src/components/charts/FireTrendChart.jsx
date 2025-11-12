import React from 'react';
import { ResponsiveLine } from '@nivo/line';
import { CHART_SERIES_COLORS, PALETTE } from './Palette';
import { useAdaptiveAxis } from './useAdaptiveAxis';

// Demo time series: wildfire counts per day
const data = [
  {
    id: 'Wildfires',
    color: CHART_SERIES_COLORS[0],
    data: [
      { x: 'Mon', y: 34 },
      { x: 'Tue', y: 28 },
      { x: 'Wed', y: 41 },
      { x: 'Thu', y: 37 },
      { x: 'Fri', y: 45 },
      { x: 'Sat', y: 39 },
      { x: 'Sun', y: 52 }
    ]
  }
];

function FireTrendChart() {
  const labels = data[0].data.map(d => d.x);
  const { ref, axisBottom } = useAdaptiveAxis(labels);
  return (
    <div className="chart-slot" ref={ref}>
      <ResponsiveLine
        data={data}
        margin={{ top: 20, right: 20, bottom: 35, left: 45 }}
        xScale={{ type: 'point' }}
        yScale={{ type: 'linear', min: 'auto', max: 'auto', stacked: false }}
        lineWidth={3}
        colors={d => d.color}
        pointSize={6}
        pointColor={{ theme: 'background' }}
        pointBorderWidth={2}
        pointBorderColor={{ from: 'serieColor' }}
        enableGridX={false}
        gridYValues={5}
        axisTop={null}
        axisRight={null}
  axisBottom={axisBottom}
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

export default FireTrendChart;
