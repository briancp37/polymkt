import { useQuery } from '@tanstack/react-query';
import { useNavigate, Link } from 'react-router-dom';
import { LineChart, Calendar, TrendingUp, TrendingDown, Clock, Plus } from 'lucide-react';
import { format } from 'date-fns';
import { listBacktests } from '../api/client';
import { Card, CardContent, Badge, Button, PageLoading, EmptyState } from '../components';
import type { BacktestSummary } from '../types';

function formatPercent(value: number | undefined): string {
  if (value === undefined) return '-';
  return `${(value * 100).toFixed(2)}%`;
}

function formatCurrency(value: number | undefined): string {
  if (value === undefined) return '-';
  const sign = value >= 0 ? '+' : '';
  return `${sign}$${value.toFixed(2)}`;
}

function StatusBadge({ status }: { status: BacktestSummary['status'] }) {
  const variants = {
    pending: 'default',
    running: 'warning',
    completed: 'success',
    failed: 'error',
  } as const;

  return <Badge variant={variants[status]}>{status}</Badge>;
}

function BacktestCard({ backtest }: { backtest: BacktestSummary }) {
  const navigate = useNavigate();
  const metrics = backtest.metrics;
  const isWinning = metrics && metrics.total_pnl > 0;

  return (
    <Card
      hoverable
      onClick={() => navigate(`/backtests/${backtest.id}`)}
      className="h-full"
    >
      <CardContent className="h-full flex flex-col">
        <div className="flex items-start justify-between mb-3">
          <div className="flex-1 min-w-0">
            <h3 className="text-lg font-semibold text-gray-900 truncate">
              {backtest.strategy_config.name}
            </h3>
            {backtest.dataset_name && (
              <p className="text-sm text-gray-500 truncate">
                on {backtest.dataset_name}
              </p>
            )}
          </div>
          <StatusBadge status={backtest.status} />
        </div>

        {/* Metrics (if completed) */}
        {metrics && (
          <div className="grid grid-cols-2 gap-3 mb-4">
            <div className="bg-gray-50 rounded-md p-2">
              <div className="text-xs text-gray-500">Total Return</div>
              <div
                className={`text-lg font-semibold ${
                  isWinning ? 'text-green-600' : 'text-red-600'
                }`}
              >
                {formatPercent(metrics.total_return)}
              </div>
            </div>
            <div className="bg-gray-50 rounded-md p-2">
              <div className="text-xs text-gray-500">Total PnL</div>
              <div
                className={`text-lg font-semibold ${
                  isWinning ? 'text-green-600' : 'text-red-600'
                }`}
              >
                {formatCurrency(metrics.total_pnl)}
              </div>
            </div>
            <div className="bg-gray-50 rounded-md p-2">
              <div className="text-xs text-gray-500">Win Rate</div>
              <div className="text-lg font-semibold text-gray-900">
                {formatPercent(metrics.win_rate)}
              </div>
            </div>
            <div className="bg-gray-50 rounded-md p-2">
              <div className="text-xs text-gray-500">Trades</div>
              <div className="text-lg font-semibold text-gray-900">
                {metrics.trade_count}
              </div>
            </div>
          </div>
        )}

        {/* Strategy details */}
        <div className="mt-auto space-y-2">
          <div className="flex items-center gap-2 text-sm text-gray-600">
            <Clock className="w-4 h-4" />
            Entry: {backtest.strategy_config.entry_days_to_exp} days to exp
          </div>
          <div className="flex items-center gap-2 text-sm text-gray-600">
            {isWinning ? (
              <TrendingUp className="w-4 h-4 text-green-600" />
            ) : (
              <TrendingDown className="w-4 h-4 text-red-600" />
            )}
            Exit: {backtest.strategy_config.exit_rule}
          </div>

          {/* Date */}
          <div className="flex items-center text-xs text-gray-400 pt-2 border-t border-gray-100">
            <Calendar className="w-3 h-3 mr-1" />
            {backtest.completed_at
              ? `Completed ${format(new Date(backtest.completed_at), 'MMM d, yyyy')}`
              : `Created ${format(new Date(backtest.created_at), 'MMM d, yyyy')}`}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export function BacktestsPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['backtests'],
    queryFn: () => listBacktests(),
  });

  if (isLoading) {
    return <PageLoading />;
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <p className="text-red-600">Failed to load backtests</p>
        <p className="text-sm text-gray-500 mt-2">
          {error instanceof Error ? error.message : 'Unknown error'}
        </p>
      </div>
    );
  }

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Backtests</h1>
          <p className="text-sm text-gray-500 mt-1">
            {data?.total_count ?? 0} backtest runs saved
          </p>
        </div>
        <Link to="/backtests/new">
          <Button>
            <Plus className="w-4 h-4 mr-2" />
            New Backtest
          </Button>
        </Link>
      </div>

      {/* Backtest Grid */}
      {data && data.items.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {data.items.map((backtest) => (
            <BacktestCard key={backtest.id} backtest={backtest} />
          ))}
        </div>
      ) : (
        <EmptyState
          icon={<LineChart className="w-12 h-12 text-gray-400" />}
          title="No backtests yet"
          description="Run your first backtest to see historical strategy performance."
          action={
            <Link to="/backtests/new">
              <Button>
                <Plus className="w-4 h-4 mr-2" />
                Create Backtest
              </Button>
            </Link>
          }
        />
      )}

      {/* Load more */}
      {data?.has_more && (
        <div className="mt-6 text-center">
          <Button variant="secondary">Load More</Button>
        </div>
      )}
    </div>
  );
}
