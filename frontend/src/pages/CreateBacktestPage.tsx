import { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useQuery, useMutation } from '@tanstack/react-query';
import { ArrowLeft, Play, Database, AlertCircle } from 'lucide-react';
import {
  listDatasets,
  prepareBacktest,
  executeBacktestSession,
} from '../api/client';
import {
  Button,
  Card,
  CardContent,
  PageLoading,
  ConfirmationModal,
  StrategyConfirmationContent,
} from '../components';
import type { DatasetSummary, StrategyConfirmation } from '../types';

const STRATEGY_EXAMPLES = [
  'Buy the favorite 90 days out, hold to expiry',
  'Buy the underdog 60 days before close with 1% fee',
  'Buy the favorite at 30 days with 2% slippage',
];

function DatasetSelector({
  datasets,
  selectedId,
  onSelect,
}: {
  datasets: DatasetSummary[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium text-gray-700">
        Select Dataset
      </label>
      <select
        value={selectedId || ''}
        onChange={(e) => onSelect(e.target.value)}
        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
      >
        <option value="">Choose a dataset...</option>
        {datasets.map((dataset) => (
          <option key={dataset.id} value={dataset.id}>
            {dataset.name} ({dataset.market_count} markets)
          </option>
        ))}
      </select>
    </div>
  );
}

export function CreateBacktestPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const preselectedDatasetId = searchParams.get('dataset');

  // Form state
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(
    preselectedDatasetId
  );
  const [strategyText, setStrategyText] = useState('');

  // Confirmation modal state
  const [showConfirmation, setShowConfirmation] = useState(false);
  const [confirmation, setConfirmation] = useState<StrategyConfirmation | null>(
    null
  );

  // Fetch datasets
  const { data: datasetsData, isLoading: datasetsLoading } = useQuery({
    queryKey: ['datasets'],
    queryFn: () => listDatasets(),
  });

  // Update selected dataset when preselected changes (e.g., from URL)
  useEffect(() => {
    if (preselectedDatasetId && !selectedDatasetId) {
      setSelectedDatasetId(preselectedDatasetId);
    }
  }, [preselectedDatasetId, selectedDatasetId]);

  // Prepare mutation (parses strategy and gets confirmation)
  const prepareMutation = useMutation({
    mutationFn: prepareBacktest,
    onSuccess: (data) => {
      setConfirmation(data);
      setShowConfirmation(true);
    },
  });

  // Execute mutation (runs the backtest after confirmation)
  const executeMutation = useMutation({
    mutationFn: executeBacktestSession,
    onSuccess: (backtest) => {
      setShowConfirmation(false);
      navigate(`/backtests/${backtest.id}`);
    },
  });

  const handlePrepare = () => {
    if (!selectedDatasetId || !strategyText.trim()) return;

    prepareMutation.mutate({
      dataset_id: selectedDatasetId,
      natural_language_strategy: strategyText.trim(),
    });
  };

  const handleConfirmExecute = () => {
    if (!confirmation) return;
    executeMutation.mutate({ session_id: confirmation.session_id });
  };

  const handleExampleClick = (example: string) => {
    setStrategyText(example);
  };

  const isFormValid = selectedDatasetId && strategyText.trim().length >= 3;
  const isPreparing = prepareMutation.isPending;
  const isExecuting = executeMutation.isPending;

  if (datasetsLoading) {
    return <PageLoading />;
  }

  const datasets = datasetsData?.datasets || [];

  return (
    <div className="max-w-2xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <button
          onClick={() => navigate('/backtests')}
          className="flex items-center text-gray-600 hover:text-gray-900 mb-4"
        >
          <ArrowLeft className="w-4 h-4 mr-1" />
          Back to Backtests
        </button>
        <h1 className="text-2xl font-bold text-gray-900">Create Backtest</h1>
        <p className="text-sm text-gray-500 mt-1">
          Describe your trading strategy in plain English and run a historical
          backtest.
        </p>
      </div>

      {/* Main Form */}
      <Card>
        <CardContent className="space-y-6">
          {/* Dataset Selector */}
          {datasets.length > 0 ? (
            <DatasetSelector
              datasets={datasets}
              selectedId={selectedDatasetId}
              onSelect={setSelectedDatasetId}
            />
          ) : (
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
              <div className="flex items-start gap-3">
                <Database className="w-5 h-5 text-amber-500 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-amber-800">
                    No datasets available
                  </p>
                  <p className="text-sm text-amber-700 mt-1">
                    You need to create a dataset first before running a
                    backtest.
                  </p>
                  <Button
                    variant="secondary"
                    className="mt-2"
                    onClick={() => navigate('/datasets/new')}
                  >
                    Create Dataset
                  </Button>
                </div>
              </div>
            </div>
          )}

          {/* Strategy Input */}
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700">
              Strategy Description
            </label>
            <textarea
              value={strategyText}
              onChange={(e) => setStrategyText(e.target.value)}
              placeholder="Describe your strategy in plain English..."
              rows={3}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none"
            />
            <p className="text-xs text-gray-500">
              Minimum 3 characters. Include entry timing, exit rules, and any
              cost assumptions.
            </p>
          </div>

          {/* Examples */}
          <div>
            <p className="text-sm text-gray-500 mb-2">Try an example:</p>
            <div className="flex flex-wrap gap-2">
              {STRATEGY_EXAMPLES.map((example, index) => (
                <button
                  key={index}
                  onClick={() => handleExampleClick(example)}
                  className="text-sm px-3 py-1.5 bg-gray-100 hover:bg-gray-200 rounded-full text-gray-700 transition-colors"
                >
                  {example}
                </button>
              ))}
            </div>
          </div>

          {/* Error Display */}
          {prepareMutation.error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <div className="flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-red-500 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-red-800">
                    Failed to parse strategy
                  </p>
                  <p className="text-sm text-red-700 mt-1">
                    {prepareMutation.error instanceof Error
                      ? prepareMutation.error.message
                      : 'An error occurred'}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Submit Button */}
          <div className="pt-4 border-t border-gray-200">
            <Button
              onClick={handlePrepare}
              disabled={!isFormValid || isPreparing}
              className="w-full"
            >
              {isPreparing ? (
                'Parsing Strategy...'
              ) : (
                <>
                  <Play className="w-4 h-4 mr-2" />
                  Preview Backtest
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Confirmation Modal */}
      <ConfirmationModal
        isOpen={showConfirmation}
        onClose={() => setShowConfirmation(false)}
        onConfirm={handleConfirmExecute}
        title="Confirm Backtest"
        confirmText="Run Backtest"
        isLoading={isExecuting}
      >
        {confirmation && (
          <StrategyConfirmationContent confirmation={confirmation} />
        )}
      </ConfirmationModal>
    </div>
  );
}
