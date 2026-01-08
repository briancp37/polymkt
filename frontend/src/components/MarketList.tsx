import { useState, useMemo } from 'react';
import { Check, ArrowUpDown, Tag, Calendar } from 'lucide-react';
import { format } from 'date-fns';
import { Badge } from './Badge';
import { Button } from './Button';
import type { MarketSearchResult } from '../types';

export interface MarketSelection {
  [marketId: string]: boolean;
}

interface MarketListProps {
  markets: MarketSearchResult[];
  selection: MarketSelection;
  onSelectionChange: (selection: MarketSelection) => void;
  hasMore?: boolean;
  onLoadMore?: () => void;
  isLoadingMore?: boolean;
  totalCount?: number;
}

type SortField = 'relevance' | 'question' | 'closed_time';
type SortDirection = 'asc' | 'desc';

export function MarketList({
  markets,
  selection,
  onSelectionChange,
  hasMore = false,
  onLoadMore,
  isLoadingMore = false,
  totalCount,
}: MarketListProps) {
  const [sortField, setSortField] = useState<SortField>('relevance');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

  const sortedMarkets = useMemo(() => {
    return [...markets].sort((a, b) => {
      let comparison = 0;
      switch (sortField) {
        case 'relevance':
          comparison = a.relevance_score - b.relevance_score;
          break;
        case 'question':
          comparison = a.question.localeCompare(b.question);
          break;
        case 'closed_time':
          const aTime = a.closed_time ? new Date(a.closed_time).getTime() : 0;
          const bTime = b.closed_time ? new Date(b.closed_time).getTime() : 0;
          comparison = aTime - bTime;
          break;
      }
      return sortDirection === 'asc' ? comparison : -comparison;
    });
  }, [markets, sortField, sortDirection]);

  const handleSort = (field: SortField) => {
    if (field === sortField) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection(field === 'relevance' ? 'desc' : 'asc');
    }
  };

  const handleToggle = (marketId: string) => {
    onSelectionChange({
      ...selection,
      [marketId]: !selection[marketId],
    });
  };

  const handleSelectAll = () => {
    const allSelected = markets.every((m) => selection[m.id]);
    const newSelection = { ...selection };
    markets.forEach((m) => {
      newSelection[m.id] = !allSelected;
    });
    onSelectionChange(newSelection);
  };

  const selectedCount = Object.values(selection).filter(Boolean).length;
  const allSelected = markets.length > 0 && markets.every((m) => selection[m.id]);

  const SortButton = ({ field, label }: { field: SortField; label: string }) => (
    <button
      onClick={() => handleSort(field)}
      className={`flex items-center gap-1 text-xs font-medium uppercase tracking-wider ${
        sortField === field ? 'text-blue-600' : 'text-gray-500 hover:text-gray-700'
      }`}
    >
      {label}
      <ArrowUpDown className="w-3 h-3" />
    </button>
  );

  if (markets.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        No markets found. Try a different search term.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Header with selection info and sort controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button
            onClick={handleSelectAll}
            className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900"
          >
            <div
              className={`w-5 h-5 rounded border flex items-center justify-center ${
                allSelected
                  ? 'bg-blue-600 border-blue-600'
                  : 'border-gray-300 bg-white'
              }`}
            >
              {allSelected && <Check className="w-3 h-3 text-white" />}
            </div>
            {allSelected ? 'Deselect All' : 'Select All'}
          </button>
          <span className="text-sm text-gray-500">
            {selectedCount} of {totalCount ?? markets.length} selected
          </span>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-xs text-gray-400">Sort by:</span>
          <SortButton field="relevance" label="Relevance" />
          <SortButton field="question" label="Name" />
          <SortButton field="closed_time" label="Close Date" />
        </div>
      </div>

      {/* Market list */}
      <div className="border border-gray-200 rounded-lg divide-y divide-gray-200">
        {sortedMarkets.map((market) => {
          const isSelected = selection[market.id] ?? true; // Default to selected
          return (
            <div
              key={market.id}
              onClick={() => handleToggle(market.id)}
              className={`flex items-start gap-3 p-4 cursor-pointer transition-colors ${
                isSelected ? 'bg-blue-50 hover:bg-blue-100' : 'bg-white hover:bg-gray-50'
              }`}
            >
              {/* Checkbox */}
              <div
                className={`mt-1 w-5 h-5 rounded border flex-shrink-0 flex items-center justify-center ${
                  isSelected
                    ? 'bg-blue-600 border-blue-600'
                    : 'border-gray-300 bg-white'
                }`}
              >
                {isSelected && <Check className="w-3 h-3 text-white" />}
              </div>

              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <h4 className="text-sm font-medium text-gray-900">
                      {market.question}
                    </h4>
                    {market.snippet && (
                      <p
                        className="text-xs text-gray-500 mt-1 line-clamp-2"
                        dangerouslySetInnerHTML={{ __html: market.snippet }}
                      />
                    )}
                  </div>
                  <Badge variant="info" className="flex-shrink-0">
                    {(market.relevance_score * 100).toFixed(1)}%
                  </Badge>
                </div>

                {/* Meta info */}
                <div className="flex items-center gap-3 mt-2">
                  {market.category && (
                    <span className="flex items-center gap-1 text-xs text-gray-500">
                      <Tag className="w-3 h-3" />
                      {market.category}
                    </span>
                  )}
                  {market.closed_time && (
                    <span className="flex items-center gap-1 text-xs text-gray-500">
                      <Calendar className="w-3 h-3" />
                      Closes {format(new Date(market.closed_time), 'MMM d, yyyy')}
                    </span>
                  )}
                  {market.tags && market.tags.length > 0 && (
                    <div className="flex gap-1">
                      {market.tags.slice(0, 3).map((tag) => (
                        <Badge key={tag} variant="default" className="text-xs">
                          {tag}
                        </Badge>
                      ))}
                      {market.tags.length > 3 && (
                        <span className="text-xs text-gray-400">
                          +{market.tags.length - 3}
                        </span>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Load more */}
      {hasMore && onLoadMore && (
        <div className="text-center pt-2">
          <Button
            variant="secondary"
            onClick={onLoadMore}
            disabled={isLoadingMore}
          >
            {isLoadingMore ? 'Loading...' : 'Load More'}
          </Button>
        </div>
      )}
    </div>
  );
}
