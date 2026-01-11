// Dataset types
export interface DatasetFilters {
  query?: string;
  category?: string;
  tags?: string[];
  closed_time_min?: string;
  closed_time_max?: string;
  min_volume?: number;
}

export interface DatasetSummary {
  id: string;
  name: string;
  description?: string;
  market_count: number;
  excluded_count: number;
  filters?: DatasetFilters;
  created_at: string;
  updated_at: string;
}

export interface Dataset extends DatasetSummary {
  market_ids: string[];
  excluded_market_ids: string[];
}

export interface DatasetListResponse {
  items: DatasetSummary[];
  count: number;
  total_count: number;
  has_more: boolean;
}

// Backtest types
export interface StrategyConfig {
  name: string;
  entry_days_to_exp: number;
  exit_rule: string;
  favorite_rule?: string;
  fee_rate: number;
  slippage_rate: number;
  position_size: number;
  extra_params?: Record<string, unknown>;
}

export interface BacktestMetrics {
  total_return: number;
  total_pnl: number;
  win_rate: number;
  trade_count: number;
  winning_trades: number;
  losing_trades: number;
  max_drawdown: number;
  sharpe_ratio?: number;
  avg_trade_pnl: number;
  avg_holding_period_days: number;
}

export interface BacktestTradeRecord {
  trade_id: string;
  election_group_id: string;
  market_id: string;
  entry_time: string;
  entry_price: number;
  exit_time?: string;
  exit_price?: number;
  position_size: number;
  gross_pnl?: number;
  fees?: number;
  slippage_cost?: number;
  net_pnl?: number;
}

export interface EquityCurvePoint {
  time: string;
  cumulative_pnl: number;
  trade_index: number;
}

export interface BacktestSummary {
  id: string;
  dataset_id: string;
  dataset_name?: string;
  strategy_config: StrategyConfig;
  status: 'pending' | 'running' | 'completed' | 'failed';
  metrics?: BacktestMetrics;
  created_at: string;
  completed_at?: string;
  error_message?: string;
}

export interface Backtest extends BacktestSummary {
  trades: BacktestTradeRecord[];
  equity_curve: EquityCurvePoint[];
}

export interface BacktestListResponse {
  items: BacktestSummary[];
  count: number;
  total_count: number;
  has_more: boolean;
}

// Market types
export interface Market {
  id: string;
  question: string;
  category?: string;
  tags?: string[];
  closed_time?: string;
  event_id?: string;
}

export interface MarketSearchResult extends Market {
  relevance_score: number;
  snippet?: string;
  bm25_score?: number;
  semantic_score?: number;
}

export interface MarketSearchResponse {
  results: MarketSearchResult[];
  count: number;
  total_count: number;
  has_more: boolean;
  mode: 'bm25' | 'semantic' | 'hybrid';
}

export interface MarketSearchParams {
  q: string;
  mode?: 'bm25' | 'semantic' | 'hybrid';
  limit?: number;
  offset?: number;
  category?: string;
  closed_time_min?: string;
  closed_time_max?: string;
}

// Strategy confirmation types
export interface ParsedStrategy {
  name: string;
  entry_days_to_exp: number;
  exit_rule: string;
  favorite_rule: string;
  fee_rate: number;
  slippage_rate: number;
  position_size: number;
  extra_params?: Record<string, unknown>;
}

export interface StrategyConfirmation {
  session_id: string;
  dataset_id: string;
  dataset_name: string;
  market_count: number;
  parsed_strategy: ParsedStrategy;
  summary: string;
  warnings: string[];
}

export interface BacktestAgentRequest {
  dataset_id: string;
  natural_language_strategy: string;
  fee_rate?: number;
  slippage_rate?: number;
  position_size?: number;
}

export interface BacktestAgentExecuteRequest {
  session_id: string;
}

// API Response types
export interface HealthResponse {
  status: string;
  version: string;
}
