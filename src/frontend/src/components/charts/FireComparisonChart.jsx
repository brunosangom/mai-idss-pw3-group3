import React from 'react';
import { ResponsiveLine } from '@nivo/line';
import { CHART_SERIES_COLORS, PALETTE } from './Palette';
import { useAdaptiveAxis } from './useAdaptiveAxis';

// Demo comparison: two series over months
const data = [
  {
    id: '2024',
    color: CHART_SERIES_COLORS[4],
    data: [
      { x: 'Jan', y: 22 }, { x: 'Feb', y: 28 }, { x: 'Mar', y: 35 }, { x: 'Apr', y: 31 },
      { x: 'May', y: 40 }, { x: 'Jun', y: 46 }, { x: 'Jul', y: 60 }, { x: 'Aug', y: 58 },
      { x: 'Sep', y: 42 }, { x: 'Oct', y: 33 }, { x: 'Nov', y: 29 }, { x: 'Dec', y: 24 }
    ]
  },
  {
    id: '2025',
    color: CHART_SERIES_COLORS[0],
    data: [
      { x: 'Jan', y: 25 }, { x: 'Feb', y: 30 }, { x: 'Mar', y: 38 }, { x: 'Apr', y: 35 },
      { x: 'May', y: 44 }, { x: 'Jun', y: 50 }, { x: 'Jul', y: 68 }, { x: 'Aug', y: 64 },
      { x: 'Sep', y: 47 }, { x: 'Oct', y: 36 }, { x: 'Nov', y: 31 }, { x: 'Dec', y: 26 }
    ]
  }
];

function FireComparisonChart() {
  // derive labels from first series (they share same x domain)
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
        pointSize={5}
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

export default FireComparisonChart;
