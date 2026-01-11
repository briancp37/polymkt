import { AlertTriangle, Clock, TrendingUp, DollarSign, Database } from 'lucide-react';
import type { StrategyConfirmation } from '../types';

interface StrategyConfirmationContentProps {
  confirmation: StrategyConfirmation;
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatExitRule(rule: string): string {
  switch (rule) {
    case 'expiry':
      return 'Hold to expiry';
    case 'take_profit':
      return 'Take profit';
    case 'stop_loss':
      return 'Stop loss';
    case 'target':
      return 'Target price';
    default:
      return rule;
  }
}

function formatFavoriteRule(rule: string): string {
  switch (rule) {
    case 'max_yes_price':
      return 'Highest YES price (favorite)';
    case 'min_yes_price':
      return 'Lowest YES price (underdog)';
    default:
      return rule;
  }
}

export function StrategyConfirmationContent({
  confirmation,
}: StrategyConfirmationContentProps) {
  const { parsed_strategy, dataset_name, market_count, summary, warnings } =
    confirmation;

  return (
    <div className="space-y-4">
      <p className="text-gray-600">
        Please review the strategy configuration before running the backtest:
      </p>

      {/* Warnings */}
      {warnings.length > 0 && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
          <div className="flex items-start gap-2">
            <AlertTriangle className="w-5 h-5 text-amber-500 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-amber-800">Warnings</p>
              <ul className="text-sm text-amber-700 mt-1 space-y-1">
                {warnings.map((warning, index) => (
                  <li key={index}>{warning}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      <div className="bg-gray-50 rounded-lg p-4 space-y-4">
        {/* Strategy name */}
        <div>
          <span className="text-sm font-medium text-gray-500">Strategy</span>
          <p className="text-gray-900 font-semibold text-lg">
            {parsed_strategy.name}
          </p>
        </div>

        {/* Dataset info */}
        <div className="flex items-center gap-2 text-gray-700">
          <Database className="w-4 h-4 text-gray-400" />
          <span>
            {dataset_name} ({market_count} markets)
          </span>
        </div>

        {/* Strategy parameters grid */}
        <div className="grid grid-cols-2 gap-3 pt-3 border-t border-gray-200">
          {/* Entry timing */}
          <div className="flex items-start gap-2">
            <Clock className="w-4 h-4 text-blue-500 mt-0.5" />
            <div>
              <span className="text-xs font-medium text-gray-500 block">
                Entry Timing
              </span>
              <span className="text-gray-900">
                {parsed_strategy.entry_days_to_exp} days to expiry
              </span>
            </div>
          </div>

          {/* Exit rule */}
          <div className="flex items-start gap-2">
            <TrendingUp className="w-4 h-4 text-green-500 mt-0.5" />
            <div>
              <span className="text-xs font-medium text-gray-500 block">
                Exit Rule
              </span>
              <span className="text-gray-900">
                {formatExitRule(parsed_strategy.exit_rule)}
              </span>
            </div>
          </div>

          {/* Favorite rule */}
          <div className="col-span-2 flex items-start gap-2">
            <TrendingUp className="w-4 h-4 text-purple-500 mt-0.5" />
            <div>
              <span className="text-xs font-medium text-gray-500 block">
                Market Selection
              </span>
              <span className="text-gray-900">
                {formatFavoriteRule(parsed_strategy.favorite_rule)}
              </span>
            </div>
          </div>
        </div>

        {/* Costs section */}
        <div className="pt-3 border-t border-gray-200">
          <div className="flex items-center gap-2 mb-2">
            <DollarSign className="w-4 h-4 text-orange-500" />
            <span className="text-sm font-medium text-gray-700">
              Trading Costs
            </span>
          </div>
          <div className="flex gap-4 text-sm">
            <div>
              <span className="text-gray-500">Fee:</span>{' '}
              <span className="text-gray-900">
                {formatPercent(parsed_strategy.fee_rate)}
              </span>
            </div>
            <div>
              <span className="text-gray-500">Slippage:</span>{' '}
              <span className="text-gray-900">
                {formatPercent(parsed_strategy.slippage_rate)}
              </span>
            </div>
            <div>
              <span className="text-gray-500">Position:</span>{' '}
              <span className="text-gray-900">
                {formatPercent(parsed_strategy.position_size)}
              </span>
            </div>
          </div>
        </div>

        {/* Summary */}
        <div className="pt-3 border-t border-gray-200">
          <span className="text-sm font-medium text-gray-500 block mb-1">
            Summary
          </span>
          <p className="text-sm text-gray-700 italic">"{summary}"</p>
        </div>
      </div>

      <p className="text-sm text-gray-500">
        Click "Run Backtest" to execute this strategy and generate results.
      </p>
    </div>
  );
}
