import { useState } from 'react';
import { Search, X } from 'lucide-react';
import { Button } from './Button';

interface MarketSearchProps {
  onSearch: (query: string, category?: string) => void;
  isLoading?: boolean;
  placeholder?: string;
}

export function MarketSearch({
  onSearch,
  isLoading = false,
  placeholder = 'Search markets (e.g., "election", "sports")...',
}: MarketSearchProps) {
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState<string>('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query.trim(), category || undefined);
    }
  };

  const handleClear = () => {
    setQuery('');
    setCategory('');
  };

  const categories = [
    { value: '', label: 'All Categories' },
    { value: 'politics', label: 'Politics' },
    { value: 'sports', label: 'Sports' },
    { value: 'crypto', label: 'Crypto' },
    { value: 'entertainment', label: 'Entertainment' },
    { value: 'science', label: 'Science' },
    { value: 'business', label: 'Business' },
  ];

  return (
    <form onSubmit={handleSubmit} className="space-y-3">
      <div className="flex gap-2">
        {/* Search input */}
        <div className="relative flex-1">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <Search className="h-5 w-5 text-gray-400" />
          </div>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="block w-full pl-10 pr-10 py-2 border border-gray-300 rounded-md leading-5 bg-white placeholder-gray-500 focus:outline-none focus:placeholder-gray-400 focus:ring-1 focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
            placeholder={placeholder}
          />
          {query && (
            <button
              type="button"
              onClick={handleClear}
              className="absolute inset-y-0 right-0 pr-3 flex items-center"
            >
              <X className="h-4 w-4 text-gray-400 hover:text-gray-600" />
            </button>
          )}
        </div>

        {/* Category filter */}
        <select
          value={category}
          onChange={(e) => setCategory(e.target.value)}
          className="block w-40 py-2 px-3 border border-gray-300 bg-white rounded-md shadow-sm focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
        >
          {categories.map((cat) => (
            <option key={cat.value} value={cat.value}>
              {cat.label}
            </option>
          ))}
        </select>

        {/* Search button */}
        <Button type="submit" disabled={isLoading || !query.trim()}>
          {isLoading ? 'Searching...' : 'Search'}
        </Button>
      </div>
    </form>
  );
}
