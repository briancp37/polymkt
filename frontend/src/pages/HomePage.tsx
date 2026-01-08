import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { Database, LineChart, TrendingUp, ArrowRight } from 'lucide-react';
import { listDatasets, listBacktests, getHealth } from '../api/client';
import { Card, CardContent, Badge, PageLoading } from '../components';

export function HomePage() {
  const { data: health, isLoading: healthLoading, error: healthError } = useQuery({
    queryKey: ['health'],
    queryFn: getHealth,
    retry: false,
  });

  const { data: datasets, isLoading: datasetsLoading } = useQuery({
    queryKey: ['datasets', 'recent'],
    queryFn: () => listDatasets(5, 0),
  });

  const { data: backtests, isLoading: backtestsLoading } = useQuery({
    queryKey: ['backtests', 'recent'],
    queryFn: () => listBacktests(5, 0),
  });

  const isLoading = healthLoading || datasetsLoading || backtestsLoading;

  if (isLoading) {
    return <PageLoading />;
  }

  const isConnected = !healthError && health?.status === 'ok';

  return (
    <div>
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Welcome to PolyMkt</h1>
        <p className="text-gray-600 mt-2">
          Backtest trading strategies on prediction markets
        </p>
      </div>

      {/* Connection Status */}
      <Card className="mb-6">
        <CardContent className="py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div
                className={`w-2 h-2 rounded-full ${
                  isConnected ? 'bg-green-500' : 'bg-red-500'
                }`}
              />
              <span className="text-sm text-gray-600">
                {isConnected ? 'Connected to API' : 'API Unavailable'}
              </span>
            </div>
            {health?.version && (
              <Badge variant="info">v{health.version}</Badge>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <Card>
          <CardContent className="py-4">
            <div className="flex items-center gap-4">
              <div className="p-3 bg-blue-100 rounded-lg">
                <Database className="w-6 h-6 text-blue-600" />
              </div>
              <div>
                <div className="text-2xl font-bold text-gray-900">
                  {datasets?.total_count ?? 0}
                </div>
                <div className="text-sm text-gray-500">Datasets</div>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="py-4">
            <div className="flex items-center gap-4">
              <div className="p-3 bg-green-100 rounded-lg">
                <LineChart className="w-6 h-6 text-green-600" />
              </div>
              <div>
                <div className="text-2xl font-bold text-gray-900">
                  {backtests?.total_count ?? 0}
                </div>
                <div className="text-sm text-gray-500">Backtests</div>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="py-4">
            <div className="flex items-center gap-4">
              <div className="p-3 bg-purple-100 rounded-lg">
                <TrendingUp className="w-6 h-6 text-purple-600" />
              </div>
              <div>
                <div className="text-2xl font-bold text-gray-900">
                  {backtests?.items.filter((b) => b.status === 'completed').length ?? 0}
                </div>
                <div className="text-sm text-gray-500">Completed Runs</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recent Activity */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Recent Datasets */}
        <Card>
          <div className="px-4 py-3 border-b border-gray-200 flex items-center justify-between">
            <h2 className="text-lg font-semibold text-gray-900">Recent Datasets</h2>
            <Link
              to="/datasets"
              className="text-sm text-blue-600 hover:text-blue-700 flex items-center"
            >
              View all <ArrowRight className="w-4 h-4 ml-1" />
            </Link>
          </div>
          <CardContent className="p-0">
            {datasets && datasets.items.length > 0 ? (
              <div className="divide-y divide-gray-100">
                {datasets.items.map((dataset) => (
                  <Link
                    key={dataset.id}
                    to={`/datasets/${dataset.id}`}
                    className="block px-4 py-3 hover:bg-gray-50"
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="font-medium text-gray-900">{dataset.name}</div>
                        <div className="text-sm text-gray-500">
                          {dataset.market_count} markets
                        </div>
                      </div>
                      <Badge variant="info">{dataset.market_count}</Badge>
                    </div>
                  </Link>
                ))}
              </div>
            ) : (
              <div className="px-4 py-8 text-center text-gray-500">
                No datasets yet
              </div>
            )}
          </CardContent>
        </Card>

        {/* Recent Backtests */}
        <Card>
          <div className="px-4 py-3 border-b border-gray-200 flex items-center justify-between">
            <h2 className="text-lg font-semibold text-gray-900">Recent Backtests</h2>
            <Link
              to="/backtests"
              className="text-sm text-blue-600 hover:text-blue-700 flex items-center"
            >
              View all <ArrowRight className="w-4 h-4 ml-1" />
            </Link>
          </div>
          <CardContent className="p-0">
            {backtests && backtests.items.length > 0 ? (
              <div className="divide-y divide-gray-100">
                {backtests.items.map((backtest) => (
                  <Link
                    key={backtest.id}
                    to={`/backtests/${backtest.id}`}
                    className="block px-4 py-3 hover:bg-gray-50"
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="font-medium text-gray-900">
                          {backtest.strategy_config.name}
                        </div>
                        <div className="text-sm text-gray-500">
                          {backtest.dataset_name ?? 'Dataset'}
                        </div>
                      </div>
                      <Badge
                        variant={
                          backtest.status === 'completed'
                            ? 'success'
                            : backtest.status === 'failed'
                            ? 'error'
                            : 'default'
                        }
                      >
                        {backtest.status}
                      </Badge>
                    </div>
                  </Link>
                ))}
              </div>
            ) : (
              <div className="px-4 py-8 text-center text-gray-500">
                No backtests yet
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
