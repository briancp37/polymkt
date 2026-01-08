import { useQuery } from '@tanstack/react-query';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { ArrowLeft, Database, Calendar, Tag, Pencil, Trash2, Play } from 'lucide-react';
import { format } from 'date-fns';
import { getDataset } from '../api/client';
import { Card, CardHeader, CardContent, Badge, Button, PageLoading } from '../components';

export function DatasetDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const { data: dataset, isLoading, error } = useQuery({
    queryKey: ['dataset', id],
    queryFn: () => getDataset(id!),
    enabled: !!id,
  });

  if (isLoading) {
    return <PageLoading />;
  }

  if (error || !dataset) {
    return (
      <div className="text-center py-12">
        <p className="text-red-600">Failed to load dataset</p>
        <p className="text-sm text-gray-500 mt-2">
          {error instanceof Error ? error.message : 'Dataset not found'}
        </p>
        <Button variant="secondary" className="mt-4" onClick={() => navigate('/datasets')}>
          Back to Datasets
        </Button>
      </div>
    );
  }

  return (
    <div>
      {/* Back navigation */}
      <Link
        to="/datasets"
        className="inline-flex items-center text-sm text-gray-500 hover:text-gray-700 mb-4"
      >
        <ArrowLeft className="w-4 h-4 mr-1" />
        Back to Datasets
      </Link>

      {/* Header */}
      <div className="flex items-start justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">{dataset.name}</h1>
          {dataset.description && (
            <p className="text-gray-600 mt-1">{dataset.description}</p>
          )}
        </div>
        <div className="flex gap-2">
          <Button variant="secondary">
            <Pencil className="w-4 h-4 mr-2" />
            Edit
          </Button>
          <Button>
            <Play className="w-4 h-4 mr-2" />
            Run Backtest
          </Button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <Card>
          <CardContent className="py-3">
            <div className="text-sm text-gray-500">Total Markets</div>
            <div className="text-2xl font-semibold text-gray-900">
              {dataset.market_count}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="py-3">
            <div className="text-sm text-gray-500">Included</div>
            <div className="text-2xl font-semibold text-green-600">
              {dataset.market_ids.length}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="py-3">
            <div className="text-sm text-gray-500">Excluded</div>
            <div className="text-2xl font-semibold text-gray-400">
              {dataset.excluded_market_ids.length}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="py-3">
            <div className="text-sm text-gray-500">Last Updated</div>
            <div className="text-lg font-semibold text-gray-900">
              {format(new Date(dataset.updated_at), 'MMM d, yyyy')}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Filters */}
      {dataset.filters && (
        <Card className="mb-6">
          <CardHeader>
            <h2 className="text-lg font-semibold text-gray-900">Applied Filters</h2>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {dataset.filters.query && (
                <Badge variant="info">
                  Query: "{dataset.filters.query}"
                </Badge>
              )}
              {dataset.filters.category && (
                <Badge variant="default">
                  <Tag className="w-3 h-3 mr-1" />
                  {dataset.filters.category}
                </Badge>
              )}
              {dataset.filters.tags?.map((tag) => (
                <Badge key={tag} variant="default">
                  {tag}
                </Badge>
              ))}
              {dataset.filters.closed_time_min && (
                <Badge variant="default">
                  <Calendar className="w-3 h-3 mr-1" />
                  From: {format(new Date(dataset.filters.closed_time_min), 'MMM d, yyyy')}
                </Badge>
              )}
              {dataset.filters.closed_time_max && (
                <Badge variant="default">
                  <Calendar className="w-3 h-3 mr-1" />
                  To: {format(new Date(dataset.filters.closed_time_max), 'MMM d, yyyy')}
                </Badge>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Market List */}
      <Card>
        <CardHeader className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-900">
            Markets ({dataset.market_ids.length})
          </h2>
        </CardHeader>
        <CardContent className="p-0">
          {dataset.market_ids.length > 0 ? (
            <div className="divide-y divide-gray-100">
              {dataset.market_ids.slice(0, 50).map((marketId) => (
                <div
                  key={marketId}
                  className="px-4 py-3 flex items-center justify-between hover:bg-gray-50"
                >
                  <div className="flex items-center">
                    <Database className="w-4 h-4 text-gray-400 mr-3" />
                    <span className="text-sm text-gray-900 font-mono">
                      {marketId}
                    </span>
                  </div>
                  <Badge variant="success">Included</Badge>
                </div>
              ))}
              {dataset.market_ids.length > 50 && (
                <div className="px-4 py-3 text-center text-sm text-gray-500">
                  + {dataset.market_ids.length - 50} more markets
                </div>
              )}
            </div>
          ) : (
            <div className="px-4 py-8 text-center text-gray-500">
              No markets in this dataset
            </div>
          )}
        </CardContent>
      </Card>

      {/* Excluded Markets */}
      {dataset.excluded_market_ids.length > 0 && (
        <Card className="mt-6">
          <CardHeader>
            <h2 className="text-lg font-semibold text-gray-900">
              Excluded Markets ({dataset.excluded_market_ids.length})
            </h2>
          </CardHeader>
          <CardContent className="p-0">
            <div className="divide-y divide-gray-100">
              {dataset.excluded_market_ids.slice(0, 10).map((marketId) => (
                <div
                  key={marketId}
                  className="px-4 py-3 flex items-center justify-between hover:bg-gray-50"
                >
                  <div className="flex items-center">
                    <Database className="w-4 h-4 text-gray-400 mr-3" />
                    <span className="text-sm text-gray-500 font-mono">
                      {marketId}
                    </span>
                  </div>
                  <Badge variant="default">Excluded</Badge>
                </div>
              ))}
              {dataset.excluded_market_ids.length > 10 && (
                <div className="px-4 py-3 text-center text-sm text-gray-500">
                  + {dataset.excluded_market_ids.length - 10} more excluded
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Danger zone */}
      <Card className="mt-6 border-red-200">
        <CardHeader className="bg-red-50 border-red-200">
          <h2 className="text-lg font-semibold text-red-700">Danger Zone</h2>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium text-gray-900">Delete this dataset</p>
              <p className="text-sm text-gray-500">
                This action cannot be undone. All backtests using this dataset will remain.
              </p>
            </div>
            <Button variant="danger">
              <Trash2 className="w-4 h-4 mr-2" />
              Delete Dataset
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
