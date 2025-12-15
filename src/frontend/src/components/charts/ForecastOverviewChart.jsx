import React from 'react';
import { ResponsiveBar } from '@nivo/bar';

const RISK_COLORS = {
  'Low': '#4caf50',
  'Moderate': '#ffeb3b',
  'High': '#ff9800',
  'Extreme': '#f44336',
  'Unknown': '#9e9e9e'
};

const ForecastOverviewChart = ({ data }) => {
  // data structure expected:
  // [
  //   { day: 'Today', Low: 10, Moderate: 5, ... },
  //   { day: '+1', Low: 8, Moderate: 7, ... },
  //   ...
  // ]

  return (
    <div style={{ height: 300 }}>
      <ResponsiveBar
        data={data}
        keys={['Low', 'Moderate', 'High', 'Extreme']}
        indexBy="day"
        margin={{ top: 50, right: 130, bottom: 50, left: 60 }}
        padding={0.3}
        valueScale={{ type: 'linear' }}
        indexScale={{ type: 'band', round: true }}
        colors={({ id }) => RISK_COLORS[id] || RISK_COLORS['Unknown']}
        borderColor={{ from: 'color', modifiers: [['darker', 1.6]] }}
        axisTop={null}
        axisRight={null}
        axisBottom={{
          tickSize: 5,
          tickPadding: 5,
          tickRotation: 0,
          legend: 'Day',
          legendPosition: 'middle',
          legendOffset: 32
        }}
        axisLeft={{
          tickSize: 5,
          tickPadding: 5,
          tickRotation: 0,
          legend: 'Number of Stations',
          legendPosition: 'middle',
          legendOffset: -40
        }}
        labelSkipWidth={12}
        labelSkipHeight={12}
        labelTextColor={{ from: 'color', modifiers: [['darker', 1.6]] }}
        legends={[
          {
            dataFrom: 'keys',
            anchor: 'bottom-right',
            direction: 'column',
            justify: false,
            translateX: 120,
            translateY: 0,
            itemsSpacing: 2,
            itemWidth: 100,
            itemHeight: 20,
            itemDirection: 'left-to-right',
            itemOpacity: 0.85,
            symbolSize: 20,
            effects: [
              {
                on: 'hover',
                style: {
                  itemOpacity: 1
                }
              }
            ]
          }
        ]}
        role="application"
        ariaLabel="Forecast Overview"
        barAriaLabel={e => `${e.id}: ${e.formattedValue} in day: ${e.indexValue}`}
      />
    </div>
  );
};

export default ForecastOverviewChart;
