import React from 'react';
import { ResponsiveLine } from '@nivo/line';

const WeatherMetricChart = ({ data, metric, title, yLabel }) => {
  if (!data || data.length === 0) return null;

  // Transform data for Nivo Line
  // data structure: [{ year, tmmx: {min, max, avg}, ... }]
  
  const chartData = [
    {
      id: "Max",
      data: data.map(d => ({ x: d.year, y: d[metric].max }))
    },
    {
      id: "Average",
      data: data.map(d => ({ x: d.year, y: d[metric].avg }))
    },
    {
      id: "Min",
      data: data.map(d => ({ x: d.year, y: d[metric].min }))
    }
  ];

  return (
    <div style={{ height: 350, width: '100%', minWidth: '100%' }}>
      <h4 style={{ textAlign: 'center', margin: '0 0 10px 0' }}>{title}</h4>
      <ResponsiveLine
        data={chartData}
        margin={{ top: 20, right: 30, bottom: 70, left: 60 }}
        xScale={{ type: 'point' }}
        yScale={{
            type: 'linear',
            min: 'auto',
            max: 'auto',
            stacked: false,
            reverse: false
        }}
        yFormat=" >-.2f"
        axisTop={null}
        axisRight={null}
        axisBottom={{
            tickSize: 5,
            tickPadding: 5,
            tickRotation: -45,
            legend: 'Year',
            legendOffset: 40,
            legendPosition: 'middle'
        }}
        axisLeft={{
            tickSize: 5,
            tickPadding: 5,
            tickRotation: 0,
            legend: yLabel,
            legendOffset: -50,
            legendPosition: 'middle'
        }}
        pointSize={8}
        pointColor={{ theme: 'background' }}
        pointBorderWidth={2}
        pointBorderColor={{ from: 'serieColor' }}
        pointLabelYOffset={-12}
        useMesh={true}
        legends={[
            {
                anchor: 'bottom',
                direction: 'row',
                justify: false,
                translateX: 0,
                translateY: 60,
                itemsSpacing: 20,
                itemDirection: 'left-to-right',
                itemWidth: 80,
                itemHeight: 20,
                itemOpacity: 0.75,
                symbolSize: 12,
                symbolShape: 'circle',
                symbolBorderColor: 'rgba(0, 0, 0, .5)',
                effects: [
                    {
                        on: 'hover',
                        style: {
                            itemBackground: 'rgba(0, 0, 0, .03)',
                            itemOpacity: 1
                        }
                    }
                ]
            }
        ]}
      />
    </div>
  );
};

export default WeatherMetricChart;
