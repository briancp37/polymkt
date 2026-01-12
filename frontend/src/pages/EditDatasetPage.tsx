import { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { ArrowLeft, Database, Check, X } from 'lucide-react';
import { getDataset, updateDataset } from '../api/client';
import {
  Card,
  CardHeader,
  CardContent,
  Button,
  PageLoading,
  ConfirmationModal,
  DatasetConfirmationContent,
} from '../components';
import type { MarketSelection } from '../components';

export function EditDatasetPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  // Form state
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [selection, setSelection] = useState<MarketSelection>({});
  const [isInitialized, setIsInitialized] = useState(false);

  // Confirmation modal state
  const [showConfirmation, setShowConfirmation] = useState(false);

  // Fetch existing dataset
  const { data: dataset, isLoading, error } = useQuery({
    queryKey: ['dataset', id],
    queryFn: () => getDataset(id!),
    enabled: !!id,
  });

  // Initialize form with dataset data
  useEffect(() => {
    if (dataset && !isInitialized) {
      setName(dataset.name);
      setDescription(dataset.description || '');

      // Initialize selection from existing market lists
      const initialSelection: MarketSelection = {};
      dataset.market_ids.forEach((marketId) => {
        initialSelection[marketId] = true;
      });
      dataset.excluded_market_ids.forEach((marketId) => {
        initialSelection[marketId] = false;
      });
      setSelection(initialSelection);
      setIsInitialized(true);
    }
  }, [dataset, isInitialized]);

  const updateDatasetMutation = useMutation({
    mutationFn: async () => {
      const selectedMarketIds = Object.entries(selection)
        .filter(([, isSelected]) => isSelected)
        .map(([marketId]) => marketId);

      const excludedMarketIds = Object.entries(selection)
        .filter(([, isSelected]) => !isSelected)
        .map(([marketId]) => marketId);

      return updateDataset(id!, {
        name,
        description: description || undefined,
        market_ids: selectedMarketIds,
        excluded_market_ids: excludedMarketIds,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['dataset', id] });
      queryClient.invalidateQueries({ queryKey: ['datasets'] });
      setShowConfirmation(false);
      navigate(`/datasets/${id}`);
    },
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

  const selectedCount = Object.values(selection).filter(Boolean).length;
  const excludedCount = Object.values(selection).filter((v) => !v).length;
  const canSave = name.trim() && selectedCount > 0;

  const handleSaveClick = () => {
    setShowConfirmation(true);
  };

  const handleConfirm = () => {
    updateDatasetMutation.mutate();
  };

  const handleCancelConfirmation = () => {
    setShowConfirmation(false);
  };

  const toggleMarket = (marketId: string) => {
    setSelection((prev) => ({
      ...prev,
      [marketId]: !prev[marketId],
    }));
  };

  const selectAll = () => {
    setSelection((prev) => {
      const updated = { ...prev };
      Object.keys(updated).forEach((id) => {
        updated[id] = true;
      });
      return updated;
    });
  };

  const deselectAll = () => {
    setSelection((prev) => {
      const updated = { ...prev };
      Object.keys(updated).forEach((id) => {
        updated[id] = false;
      });
      return updated;
    });
  };

  // Get all market IDs from both included and excluded
  const allMarketIds = [
    ...dataset.market_ids,
    ...dataset.excluded_market_ids,
  ];

  return (
    <div>
      {/* Back navigation */}
      <Link
        to={`/datasets/${id}`}
        className="inline-flex items-center text-sm text-gray-500 hover:text-gray-700 mb-4"
      >
        <ArrowLeft className="w-4 h-4 mr-1" />
        Back to Dataset
      </Link>

      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Edit Dataset</h1>
        <p className="text-gray-600 mt-1">
          Modify the name, description, or market selections for this dataset
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main content - Market List */}
        <div className="lg:col-span-2 space-y-6">
          {/* Market List */}
          <Card>
            <CardHeader className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-gray-900">
                Markets ({allMarketIds.length})
              </h2>
              <div className="flex gap-2">
                <Button variant="secondary" size="sm" onClick={selectAll}>
                  <Check className="w-4 h-4 mr-1" />
                  Select All
                </Button>
                <Button variant="secondary" size="sm" onClick={deselectAll}>
                  <X className="w-4 h-4 mr-1" />
                  Deselect All
                </Button>
              </div>
            </CardHeader>
            <CardContent className="p-0">
              {allMarketIds.length > 0 ? (
                <div className="divide-y divide-gray-100 max-h-[600px] overflow-y-auto">
                  {allMarketIds.map((marketId) => {
                    const isSelected = selection[marketId] ?? true;
                    return (
                      <div
                        key={marketId}
                        className={`px-4 py-3 flex items-center justify-between hover:bg-gray-50 cursor-pointer ${
                          !isSelected ? 'bg-gray-50' : ''
                        }`}
                        onClick={() => toggleMarket(marketId)}
                      >
                        <div className="flex items-center">
                          <input
                            type="checkbox"
                            checked={isSelected}
                            onChange={() => toggleMarket(marketId)}
                            className="h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500 mr-3"
                            onClick={(e) => e.stopPropagation()}
                          />
                          <Database className="w-4 h-4 text-gray-400 mr-3" />
                          <span
                            className={`text-sm font-mono ${
                              isSelected ? 'text-gray-900' : 'text-gray-400'
                            }`}
                          >
                            {marketId}
                          </span>
                        </div>
                        <span
                          className={`text-xs px-2 py-1 rounded ${
                            isSelected
                              ? 'bg-green-100 text-green-700'
                              : 'bg-gray-100 text-gray-500'
                          }`}
                        >
                          {isSelected ? 'Included' : 'Excluded'}
                        </span>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="px-4 py-8 text-center text-gray-500">
                  No markets in this dataset
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Sidebar - Dataset info and save */}
        <div className="space-y-6">
          {/* Dataset form */}
          <Card>
            <CardHeader>
              <h2 className="text-lg font-semibold text-gray-900">
                Dataset Details
              </h2>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label
                  htmlFor="name"
                  className="block text-sm font-medium text-gray-700 mb-1"
                >
                  Name *
                </label>
                <input
                  type="text"
                  id="name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                  placeholder="e.g., Election Markets 2024"
                />
              </div>
              <div>
                <label
                  htmlFor="description"
                  className="block text-sm font-medium text-gray-700 mb-1"
                >
                  Description
                </label>
                <textarea
                  id="description"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  rows={3}
                  className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                  placeholder="Optional description..."
                />
              </div>
            </CardContent>
          </Card>

          {/* Selection summary */}
          <Card>
            <CardHeader>
              <h2 className="text-lg font-semibold text-gray-900">
                Selection Summary
              </h2>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex justify-between text-sm">
                <span className="text-gray-500">Included Markets</span>
                <span className="font-medium text-green-600">{selectedCount}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-500">Excluded Markets</span>
                <span className="font-medium text-gray-400">
                  {excludedCount}
                </span>
              </div>
              {dataset.filters?.query && (
                <div className="pt-2 border-t border-gray-100">
                  <span className="text-xs text-gray-400">
                    Original search: "{dataset.filters.query}"
                  </span>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Save button */}
          <div className="space-y-3">
            <Button
              className="w-full"
              size="lg"
              disabled={!canSave}
              onClick={handleSaveClick}
            >
              Save Changes
            </Button>
            <Button
              variant="secondary"
              className="w-full"
              onClick={() => navigate(`/datasets/${id}`)}
            >
              Cancel
            </Button>
          </div>

          {updateDatasetMutation.isError && (
            <p className="text-sm text-red-600 text-center">
              Failed to update dataset. Please try again.
            </p>
          )}
        </div>
      </div>

      {/* Confirmation Modal */}
      <ConfirmationModal
        isOpen={showConfirmation}
        onClose={handleCancelConfirmation}
        onConfirm={handleConfirm}
        title="Confirm Dataset Update"
        confirmText="Save Changes"
        isLoading={updateDatasetMutation.isPending}
      >
        <DatasetConfirmationContent
          name={name}
          description={description || undefined}
          includedCount={selectedCount}
          excludedCount={excludedCount}
          searchQuery={dataset.filters?.query}
          searchCategory={dataset.filters?.category}
        />
      </ConfirmationModal>
    </div>
  );
}
