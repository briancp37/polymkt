import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { format } from 'date-fns';
import type { EquityCurvePoint } from '../types';

interface EquityCurveChartProps {
  data: EquityCurvePoint[];
  className?: string;
}

function formatCurrency(value: number): string {
  const sign = value >= 0 ? '+' : '';
  return `${sign}$${value.toFixed(2)}`;
}

interface TooltipPayloadItem {
  value: number;
  dataKey: string;
  payload: EquityCurvePoint;
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: TooltipPayloadItem[];
  label?: string;
}

function CustomTooltip({ active, payload }: CustomTooltipProps) {
  if (!active || !payload || !payload[0]) {
    return null;
  }

  const point = payload[0].payload;
  const isPositive = point.cumulative_pnl >= 0;

  return (
    <div className="bg-white border border-gray-200 rounded shadow-lg p-3">
      <p className="text-sm text-gray-500 mb-1">
        {format(new Date(point.time), 'MMM d, yyyy HH:mm')}
      </p>
      <p className={`text-lg font-semibold ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
        {formatCurrency(point.cumulative_pnl)}
      </p>
      <p className="text-xs text-gray-400">
        Trade #{point.trade_index}
      </p>
    </div>
  );
}

export function EquityCurveChart({ data, className }: EquityCurveChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className={`h-64 bg-gray-50 rounded flex items-center justify-center text-gray-400 ${className || ''}`}>
        <p>No equity curve data available</p>
      </div>
    );
  }

  // Determine min/max for Y axis padding
  const pnlValues = data.map((d) => d.cumulative_pnl);
  const minPnl = Math.min(...pnlValues, 0);
  const maxPnl = Math.max(...pnlValues, 0);
  const padding = Math.max(Math.abs(maxPnl - minPnl) * 0.1, 10);

  // Format X axis dates
  const formatXAxis = (time: string) => {
    return format(new Date(time), 'MMM d');
  };

  // Determine line color based on final PnL
  const finalPnl = data[data.length - 1].cumulative_pnl;
  const lineColor = finalPnl >= 0 ? '#16a34a' : '#dc2626'; // green-600 or red-600

  return (
    <div className={`h-64 ${className || ''}`}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={data}
          margin={{ top: 10, right: 10, left: 10, bottom: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis
            dataKey="time"
            tickFormatter={formatXAxis}
            stroke="#9ca3af"
            fontSize={12}
            tickLine={false}
            axisLine={{ stroke: '#e5e7eb' }}
          />
          <YAxis
            tickFormatter={(value: number) => `$${value.toFixed(0)}`}
            stroke="#9ca3af"
            fontSize={12}
            tickLine={false}
            axisLine={{ stroke: '#e5e7eb' }}
            domain={[minPnl - padding, maxPnl + padding]}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine y={0} stroke="#9ca3af" strokeDasharray="3 3" />
          <Line
            type="monotone"
            dataKey="cumulative_pnl"
            stroke={lineColor}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: lineColor }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
