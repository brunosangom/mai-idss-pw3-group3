import React from 'react';
import { ResponsiveLine } from '@nivo/line';
import { CHART_SERIES_COLORS, PALETTE } from './Palette';
import { useAdaptiveAxis } from './useAdaptiveAxis';

const data = [
  {
    id: 'Active',
    color: CHART_SERIES_COLORS[1], // orange
    data: [
      { x: 'Jan', y: 10 }, { x: 'Feb', y: 12 }, { x: 'Mar', y: 18 }, { x: 'Apr', y: 25 },
      { x: 'May', y: 35 }, { x: 'Jun', y: 48 }, { x: 'Jul', y: 62 }, { x: 'Aug', y: 58 },
      { x: 'Sep', y: 40 }, { x: 'Oct', y: 28 }, { x: 'Nov', y: 18 }, { x: 'Dec', y: 12 }
    ]
  },
  {
    id: 'Contained',
    color: CHART_SERIES_COLORS[3], // green
    data: [
      { x: 'Jan', y: 8 }, { x: 'Feb', y: 9 }, { x: 'Mar', y: 12 }, { x: 'Apr', y: 18 },
      { x: 'May', y: 24 }, { x: 'Jun', y: 30 }, { x: 'Jul', y: 44 }, { x: 'Aug', y: 46 },
      { x: 'Sep', y: 35 }, { x: 'Oct', y: 24 }, { x: 'Nov', y: 16 }, { x: 'Dec', y: 10 }
    ]
  }
];

function SeasonalityChart() {
  const labels = data[0].data.map(d => d.x);
  const { ref, axisBottom } = useAdaptiveAxis(labels);
  return (
    <div className="chart-slot" ref={ref}>
      <ResponsiveLine
        data={data}
        margin={{ top: 20, right: 20, bottom: 35, left: 45 }}
        xScale={{ type: 'point' }}
        yScale={{ type: 'linear', min: 'auto', max: 'auto', stacked: false }}
        enableArea={true}
        areaOpacity={0.15}
        colors={d => d.color}
        lineWidth={3}
        pointSize={5}
        pointBorderWidth={2}
        pointColor={{ theme: 'background' }}
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

export default SeasonalityChart;
