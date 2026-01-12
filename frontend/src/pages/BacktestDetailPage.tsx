import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import {
  ArrowLeft,
  TrendingUp,
  TrendingDown,
  Calendar,
  Clock,
  DollarSign,
  Percent,
  BarChart2,
  Download,
  RefreshCw,
  Trash2,
} from 'lucide-react';
import { format } from 'date-fns';
import { getBacktest } from '../api/client';
import { Card, CardHeader, CardContent, Badge, Button, PageLoading, EquityCurveChart } from '../components';
import type { BacktestTradeRecord, BacktestMetrics } from '../types';

const TRADES_PER_PAGE = 20;

function formatPercent(value: number | undefined): string {
  if (value === undefined) return '-';
  return `${(value * 100).toFixed(2)}%`;
}

function formatCurrency(value: number | undefined): string {
  if (value === undefined) return '-';
  const sign = value >= 0 ? '+' : '';
  return `${sign}$${value.toFixed(2)}`;
}

function StatusBadge({ status }: { status: string }) {
  const variants = {
    pending: 'default',
    running: 'warning',
    completed: 'success',
    failed: 'error',
  } as const;

  return (
    <Badge variant={variants[status as keyof typeof variants] || 'default'}>
      {status}
    </Badge>
  );
}

function MetricCard({
  label,
  value,
  icon: Icon,
  trend,
}: {
  label: string;
  value: string;
  icon: React.ComponentType<{ className?: string }>;
  trend?: 'up' | 'down' | 'neutral';
}) {
  const trendColors = {
    up: 'text-green-600',
    down: 'text-red-600',
    neutral: 'text-gray-900',
  };

  return (
    <Card>
      <CardContent className="py-3">
        <div className="flex items-center gap-2 text-sm text-gray-500 mb-1">
          <Icon className="w-4 h-4" />
          {label}
        </div>
        <div className={`text-2xl font-semibold ${trendColors[trend || 'neutral']}`}>
          {value}
        </div>
      </CardContent>
    </Card>
  );
}

function formatSharpeRatio(value: number | undefined): string {
  if (value === undefined || value === null) return '-';
  return value.toFixed(2);
}

function MetricsGrid({ metrics }: { metrics: BacktestMetrics }) {
  const isWinning = metrics.total_pnl > 0;
  const sharpeRatio = metrics.sharpe_ratio;
  const sharpeTrend = sharpeRatio !== undefined && sharpeRatio !== null
    ? (sharpeRatio > 1 ? 'up' : sharpeRatio < 0 ? 'down' : 'neutral')
    : 'neutral';

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <MetricCard
        label="Total Return"
        value={formatPercent(metrics.total_return)}
        icon={Percent}
        trend={isWinning ? 'up' : 'down'}
      />
      <MetricCard
        label="Total PnL"
        value={formatCurrency(metrics.total_pnl)}
        icon={DollarSign}
        trend={isWinning ? 'up' : 'down'}
      />
      <MetricCard
        label="Win Rate"
        value={formatPercent(metrics.win_rate)}
        icon={BarChart2}
        trend={metrics.win_rate > 0.5 ? 'up' : 'down'}
      />
      <MetricCard
        label="Sharpe Ratio"
        value={formatSharpeRatio(metrics.sharpe_ratio)}
        icon={TrendingUp}
        trend={sharpeTrend}
      />
      <MetricCard
        label="Trade Count"
        value={metrics.trade_count.toString()}
        icon={BarChart2}
      />
      <MetricCard
        label="Winning Trades"
        value={metrics.winning_trades.toString()}
        icon={TrendingUp}
        trend="up"
      />
      <MetricCard
        label="Losing Trades"
        value={metrics.losing_trades.toString()}
        icon={TrendingDown}
        trend={metrics.losing_trades > 0 ? 'down' : 'neutral'}
      />
      <MetricCard
        label="Max Drawdown"
        value={formatPercent(metrics.max_drawdown)}
        icon={TrendingDown}
        trend="down"
      />
      <MetricCard
        label="Avg PnL/Trade"
        value={formatCurrency(metrics.avg_trade_pnl)}
        icon={DollarSign}
        trend={metrics.avg_trade_pnl > 0 ? 'up' : 'down'}
      />
      <MetricCard
        label="Avg Holding Days"
        value={metrics.avg_holding_period_days.toFixed(1)}
        icon={Clock}
      />
    </div>
  );
}

interface TradesTableProps {
  trades: BacktestTradeRecord[];
  page: number;
  onPageChange: (page: number) => void;
}

function TradesTable({ trades, page, onPageChange }: TradesTableProps) {
  const totalPages = Math.ceil(trades.length / TRADES_PER_PAGE);
  const startIndex = page * TRADES_PER_PAGE;
  const endIndex = Math.min(startIndex + TRADES_PER_PAGE, trades.length);
  const paginatedTrades = trades.slice(startIndex, endIndex);

  return (
    <div>
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Group / Market
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Entry
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Exit
              </th>
              <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                PnL
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {paginatedTrades.map((trade) => {
              const isWinner = (trade.net_pnl ?? 0) > 0;
              return (
                <tr key={trade.trade_id} className="hover:bg-gray-50">
                  <td className="px-4 py-3">
                    <div className="text-sm font-medium text-gray-900 truncate max-w-[200px]">
                      {trade.election_group_id}
                    </div>
                    <div className="text-xs text-gray-500 font-mono truncate max-w-[200px]">
                      {trade.market_id}
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <div className="text-sm text-gray-900">
                      ${trade.entry_price.toFixed(3)}
                    </div>
                    <div className="text-xs text-gray-500">
                      {format(new Date(trade.entry_time), 'MMM d, yyyy')}
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    {trade.exit_time ? (
                      <>
                        <div className="text-sm text-gray-900">
                          ${trade.exit_price?.toFixed(3) ?? '-'}
                        </div>
                        <div className="text-xs text-gray-500">
                          {format(new Date(trade.exit_time), 'MMM d, yyyy')}
                        </div>
                      </>
                    ) : (
                      <span className="text-sm text-gray-400">Open</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <div
                      className={`text-sm font-medium ${
                        isWinner ? 'text-green-600' : 'text-red-600'
                      }`}
                    >
                      {formatCurrency(trade.net_pnl)}
                    </div>
                    {trade.fees && trade.fees > 0 && (
                      <div className="text-xs text-gray-400">
                        Fees: ${trade.fees.toFixed(4)}
                      </div>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between px-4 py-3 border-t border-gray-200">
          <div className="text-sm text-gray-500">
            Showing {startIndex + 1}-{endIndex} of {trades.length} trades
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => onPageChange(page - 1)}
              disabled={page === 0}
            >
              <ChevronLeft className="w-4 h-4" />
              Previous
            </Button>
            <span className="text-sm text-gray-500">
              Page {page + 1} of {totalPages}
            </span>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => onPageChange(page + 1)}
              disabled={page >= totalPages - 1}
            >
              Next
              <ChevronRight className="w-4 h-4" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

export function BacktestDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [tradesPage, setTradesPage] = useState(0);

  const { data: backtest, isLoading, error } = useQuery({
    queryKey: ['backtest', id],
    queryFn: () => getBacktest(id!),
    enabled: !!id,
  });

  if (isLoading) {
    return <PageLoading />;
  }

  if (error || !backtest) {
    return (
      <div className="text-center py-12">
        <p className="text-red-600">Failed to load backtest</p>
        <p className="text-sm text-gray-500 mt-2">
          {error instanceof Error ? error.message : 'Backtest not found'}
        </p>
        <Button variant="secondary" className="mt-4" onClick={() => navigate('/backtests')}>
          Back to Backtests
        </Button>
      </div>
    );
  }

  const isCompleted = backtest.status === 'completed';
  const isFailed = backtest.status === 'failed';

  return (
    <div>
      {/* Back navigation */}
      <Link
        to="/backtests"
        className="inline-flex items-center text-sm text-gray-500 hover:text-gray-700 mb-4"
      >
        <ArrowLeft className="w-4 h-4 mr-1" />
        Back to Backtests
      </Link>

      {/* Header */}
      <div className="flex items-start justify-between mb-6">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-gray-900">
              {backtest.strategy_config.name}
            </h1>
            <StatusBadge status={backtest.status} />
          </div>
          {backtest.dataset_name && (
            <p className="text-gray-600 mt-1">Dataset: {backtest.dataset_name}</p>
          )}
        </div>
        <div className="flex gap-2">
          <Button variant="secondary">
            <Download className="w-4 h-4 mr-2" />
            Export
          </Button>
          <Button variant="secondary">
            <RefreshCw className="w-4 h-4 mr-2" />
            Rerun
          </Button>
        </div>
      </div>

      {/* Error message */}
      {isFailed && backtest.error_message && (
        <Card className="mb-6 border-red-200 bg-red-50">
          <CardContent>
            <p className="text-red-700">{backtest.error_message}</p>
          </CardContent>
        </Card>
      )}

      {/* Strategy Config */}
      <Card className="mb-6">
        <CardHeader>
          <h2 className="text-lg font-semibold text-gray-900">Strategy Configuration</h2>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-gray-500">Entry Days to Exp:</span>
              <span className="ml-2 font-medium">
                {backtest.strategy_config.entry_days_to_exp}
              </span>
            </div>
            <div>
              <span className="text-gray-500">Exit Rule:</span>
              <span className="ml-2 font-medium">{backtest.strategy_config.exit_rule}</span>
            </div>
            <div>
              <span className="text-gray-500">Fee Rate:</span>
              <span className="ml-2 font-medium">
                {(backtest.strategy_config.fee_rate * 100).toFixed(2)}%
              </span>
            </div>
            <div>
              <span className="text-gray-500">Slippage:</span>
              <span className="ml-2 font-medium">
                {(backtest.strategy_config.slippage_rate * 100).toFixed(2)}%
              </span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Metrics (if completed) */}
      {isCompleted && backtest.metrics && (
        <>
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Key Metrics</h2>
          <div className="mb-6">
            <MetricsGrid metrics={backtest.metrics} />
          </div>
        </>
      )}

      {/* Equity Curve Chart */}
      {isCompleted && backtest.equity_curve && backtest.equity_curve.length > 0 && (
        <Card className="mb-6">
          <CardHeader>
            <h2 className="text-lg font-semibold text-gray-900">Equity Curve</h2>
          </CardHeader>
          <CardContent>
            <EquityCurveChart data={backtest.equity_curve} />
          </CardContent>
        </Card>
      )}

      {/* Trades Table (if completed) */}
      {isCompleted && backtest.trades && backtest.trades.length > 0 && (
        <Card className="mb-6">
          <CardHeader className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-gray-900">
              Trades ({backtest.trades.length})
            </h2>
            <Button variant="ghost" size="sm">
              <Download className="w-4 h-4 mr-2" />
              Export CSV
            </Button>
          </CardHeader>
          <CardContent className="p-0">
            <TradesTable
              trades={backtest.trades}
              page={tradesPage}
              onPageChange={setTradesPage}
            />
          </CardContent>
        </Card>
      )}

      {/* Timestamps */}
      <Card className="mb-6">
        <CardContent className="py-3">
          <div className="flex items-center gap-6 text-sm text-gray-500">
            <div className="flex items-center gap-1">
              <Calendar className="w-4 h-4" />
              Created: {format(new Date(backtest.created_at), 'MMM d, yyyy HH:mm')}
            </div>
            {backtest.completed_at && (
              <div className="flex items-center gap-1">
                <Clock className="w-4 h-4" />
                Completed: {format(new Date(backtest.completed_at), 'MMM d, yyyy HH:mm')}
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Danger zone */}
      <Card className="border-red-200">
        <CardHeader className="bg-red-50 border-red-200">
          <h2 className="text-lg font-semibold text-red-700">Danger Zone</h2>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium text-gray-900">Delete this backtest</p>
              <p className="text-sm text-gray-500">
                This action cannot be undone.
              </p>
            </div>
            <Button variant="danger">
              <Trash2 className="w-4 h-4 mr-2" />
              Delete Backtest
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
