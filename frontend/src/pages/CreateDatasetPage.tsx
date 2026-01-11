import { useState, useCallback } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useMutation } from '@tanstack/react-query';
import { ArrowLeft, Database, Search as SearchIcon } from 'lucide-react';
import { searchMarkets, createDataset } from '../api/client';
import {
  Card,
  CardHeader,
  CardContent,
  Button,
  MarketSearch,
  MarketList,
  LoadingSpinner,
  EmptyState,
  ConfirmationModal,
  DatasetConfirmationContent,
} from '../components';
import type { MarketSelection } from '../components';
import type { MarketSearchResult, DatasetFilters } from '../types';

export function CreateDatasetPage() {
  const navigate = useNavigate();

  // Search state
  const [searchQuery, setSearchQuery] = useState('');
  const [searchCategory, setSearchCategory] = useState<string | undefined>();
  const [markets, setMarkets] = useState<MarketSearchResult[]>([]);
  const [selection, setSelection] = useState<MarketSelection>({});
  const [hasMore, setHasMore] = useState(false);
  const [totalCount, setTotalCount] = useState(0);
  const [offset, setOffset] = useState(0);
  const [isSearching, setIsSearching] = useState(false);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);

  // Dataset form state
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');

  // Confirmation modal state
  const [showConfirmation, setShowConfirmation] = useState(false);

  const LIMIT = 50;

  const handleSearch = useCallback(async (query: string, category?: string) => {
    setIsSearching(true);
    setSearchQuery(query);
    setSearchCategory(category);
    setHasSearched(true);

    try {
      const response = await searchMarkets({
        q: query,
        category,
        limit: LIMIT,
        offset: 0,
        mode: 'hybrid',
      });

      setMarkets(response.results);
      setHasMore(response.has_more);
      setTotalCount(response.total_count);
      setOffset(LIMIT);

      // Initialize selection - all selected by default
      const initialSelection: MarketSelection = {};
      response.results.forEach((m) => {
        initialSelection[m.id] = true;
      });
      setSelection(initialSelection);
    } catch (error) {
      console.error('Search failed:', error);
    } finally {
      setIsSearching(false);
    }
  }, []);

  const handleLoadMore = useCallback(async () => {
    if (!searchQuery || isLoadingMore) return;

    setIsLoadingMore(true);
    try {
      const response = await searchMarkets({
        q: searchQuery,
        category: searchCategory,
        limit: LIMIT,
        offset,
        mode: 'hybrid',
      });

      setMarkets((prev) => [...prev, ...response.results]);
      setHasMore(response.has_more);
      setOffset((prev) => prev + LIMIT);

      // Add new markets to selection (selected by default)
      setSelection((prev) => {
        const updated = { ...prev };
        response.results.forEach((m) => {
          if (!(m.id in updated)) {
            updated[m.id] = true;
          }
        });
        return updated;
      });
    } catch (error) {
      console.error('Load more failed:', error);
    } finally {
      setIsLoadingMore(false);
    }
  }, [searchQuery, searchCategory, offset, isLoadingMore]);

  const createDatasetMutation = useMutation({
    mutationFn: async () => {
      const selectedMarketIds = Object.entries(selection)
        .filter(([, isSelected]) => isSelected)
        .map(([id]) => id);

      const excludedMarketIds = Object.entries(selection)
        .filter(([, isSelected]) => !isSelected)
        .map(([id]) => id);

      const filters: DatasetFilters = {};
      if (searchQuery) filters.query = searchQuery;
      if (searchCategory) filters.category = searchCategory;

      return createDataset({
        name,
        description: description || undefined,
        market_ids: selectedMarketIds,
        excluded_market_ids: excludedMarketIds,
        filters,
      });
    },
    onSuccess: (dataset) => {
      setShowConfirmation(false);
      navigate(`/datasets/${dataset.id}`);
    },
  });

  const selectedCount = Object.values(selection).filter(Boolean).length;
  const excludedCount = Object.values(selection).filter((v) => !v).length;
  const canSave = name.trim() && selectedCount > 0;

  const handleCreateClick = () => {
    setShowConfirmation(true);
  };

  const handleConfirm = () => {
    createDatasetMutation.mutate();
  };

  const handleCancelConfirmation = () => {
    setShowConfirmation(false);
  };

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
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Create Dataset</h1>
        <p className="text-gray-600 mt-1">
          Search for markets and select which ones to include in your dataset
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main content - Search and Market List */}
        <div className="lg:col-span-2 space-y-6">
          {/* Search */}
          <Card>
            <CardHeader>
              <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
                <SearchIcon className="w-5 h-5" />
                Search Markets
              </h2>
            </CardHeader>
            <CardContent>
              <MarketSearch onSearch={handleSearch} isLoading={isSearching} />
            </CardContent>
          </Card>

          {/* Market List */}
          <Card>
            <CardHeader>
              <h2 className="text-lg font-semibold text-gray-900">
                Market List
                {totalCount > 0 && (
                  <span className="ml-2 text-sm font-normal text-gray-500">
                    ({totalCount} results)
                  </span>
                )}
              </h2>
            </CardHeader>
            <CardContent>
              {isSearching ? (
                <div className="flex items-center justify-center py-12">
                  <LoadingSpinner size="lg" />
                </div>
              ) : hasSearched && markets.length > 0 ? (
                <MarketList
                  markets={markets}
                  selection={selection}
                  onSelectionChange={setSelection}
                  hasMore={hasMore}
                  onLoadMore={handleLoadMore}
                  isLoadingMore={isLoadingMore}
                  totalCount={totalCount}
                />
              ) : hasSearched ? (
                <EmptyState
                  icon={<Database className="w-12 h-12 text-gray-400" />}
                  title="No markets found"
                  description="Try a different search term or adjust your filters."
                />
              ) : (
                <EmptyState
                  icon={<SearchIcon className="w-12 h-12 text-gray-400" />}
                  title="Search for markets"
                  description="Enter a search term above to find markets to include in your dataset."
                />
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
              {searchQuery && (
                <div className="pt-2 border-t border-gray-100">
                  <span className="text-xs text-gray-400">Search: "{searchQuery}"</span>
                  {searchCategory && (
                    <span className="text-xs text-gray-400 ml-2">
                      Category: {searchCategory}
                    </span>
                  )}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Save button */}
          <Button
            className="w-full"
            size="lg"
            disabled={!canSave}
            onClick={handleCreateClick}
          >
            Create Dataset
          </Button>

          {createDatasetMutation.isError && (
            <p className="text-sm text-red-600 text-center">
              Failed to create dataset. Please try again.
            </p>
          )}
        </div>
      </div>

      {/* Confirmation Modal */}
      <ConfirmationModal
        isOpen={showConfirmation}
        onClose={handleCancelConfirmation}
        onConfirm={handleConfirm}
        title="Confirm Dataset Creation"
        confirmText="Create Dataset"
        isLoading={createDatasetMutation.isPending}
      >
        <DatasetConfirmationContent
          name={name}
          description={description || undefined}
          includedCount={selectedCount}
          excludedCount={excludedCount}
          searchQuery={searchQuery || undefined}
          searchCategory={searchCategory}
        />
      </ConfirmationModal>
    </div>
  );
}
