import { useQuery } from '@tanstack/react-query';
import { useNavigate, Link } from 'react-router-dom';
import { Database, Calendar, Tag, Plus } from 'lucide-react';
import { format } from 'date-fns';
import { listDatasets } from '../api/client';
import { Card, CardContent, Badge, Button, PageLoading, EmptyState } from '../components';
import type { DatasetSummary } from '../types';

function DatasetCard({ dataset }: { dataset: DatasetSummary }) {
  const navigate = useNavigate();

  return (
    <Card
      hoverable
      onClick={() => navigate(`/datasets/${dataset.id}`)}
      className="h-full"
    >
      <CardContent className="h-full flex flex-col">
        <div className="flex items-start justify-between mb-3">
          <h3 className="text-lg font-semibold text-gray-900 truncate flex-1">
            {dataset.name}
          </h3>
          <Badge variant="info" className="ml-2 shrink-0">
            {dataset.market_count} markets
          </Badge>
        </div>

        {dataset.description && (
          <p className="text-sm text-gray-600 mb-4 line-clamp-2">
            {dataset.description}
          </p>
        )}

        <div className="mt-auto space-y-2">
          {/* Stats row */}
          <div className="flex items-center gap-4 text-sm text-gray-500">
            <span className="flex items-center gap-1">
              <Database className="w-4 h-4" />
              {dataset.market_count} included
            </span>
            {dataset.excluded_count > 0 && (
              <span className="text-gray-400">
                ({dataset.excluded_count} excluded)
              </span>
            )}
          </div>

          {/* Filters/Tags */}
          {dataset.filters && (
            <div className="flex flex-wrap gap-1">
              {dataset.filters.category && (
                <Badge variant="default">
                  <Tag className="w-3 h-3 mr-1" />
                  {dataset.filters.category}
                </Badge>
              )}
              {dataset.filters.tags?.slice(0, 2).map((tag) => (
                <Badge key={tag} variant="default">
                  {tag}
                </Badge>
              ))}
              {dataset.filters.tags && dataset.filters.tags.length > 2 && (
                <Badge variant="default">
                  +{dataset.filters.tags.length - 2} more
                </Badge>
              )}
            </div>
          )}

          {/* Date range */}
          <div className="flex items-center text-xs text-gray-400 pt-2 border-t border-gray-100">
            <Calendar className="w-3 h-3 mr-1" />
            Updated {format(new Date(dataset.updated_at), 'MMM d, yyyy')}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export function DatasetsPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['datasets'],
    queryFn: () => listDatasets(),
  });

  if (isLoading) {
    return <PageLoading />;
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <p className="text-red-600">Failed to load datasets</p>
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
          <h1 className="text-2xl font-bold text-gray-900">Datasets</h1>
          <p className="text-sm text-gray-500 mt-1">
            {data?.total_count ?? 0} datasets saved
          </p>
        </div>
        <Link to="/datasets/new">
          <Button>
            <Plus className="w-4 h-4 mr-2" />
            New Dataset
          </Button>
        </Link>
      </div>

      {/* Dataset Grid */}
      {data && data.items.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {data.items.map((dataset) => (
            <DatasetCard key={dataset.id} dataset={dataset} />
          ))}
        </div>
      ) : (
        <EmptyState
          icon={<Database className="w-12 h-12 text-gray-400" />}
          title="No datasets yet"
          description="Create your first dataset to start organizing markets for backtesting."
          action={
            <Link to="/datasets/new">
              <Button>
                <Plus className="w-4 h-4 mr-2" />
                Create Dataset
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
