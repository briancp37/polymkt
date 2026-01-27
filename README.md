# PolyMkt

A professional DuckDB-over-Parquet analytics stack for Polymarket prediction market data, featuring backtesting, search, and real-time wallet tracking.

## Overview

PolyMkt is a comprehensive analytics platform for analyzing Polymarket prediction market data. It provides:

- **Data Pipeline**: CSV to Parquet conversion with validation, partitioning, and incremental updates
- **Search**: Hybrid search combining BM25 full-text and semantic search (OpenAI embeddings)
- **Backtesting**: Election group-based strategies with the "buy the favorite" system
- **Wallet Tracking**: Sharp money monitoring with average-cost accounting and mark-to-market P&L
- **React Frontend**: Modern UI for datasets, backtests, and market exploration
- **LLM Analytics**: Guarded analytics interface for AI-driven queries

## Tech Stack

### Backend
- **Python 3.11+** - Core language
- **FastAPI** - REST API framework
- **DuckDB** - Embedded analytics database
- **PyArrow** - Parquet file operations
- **Pydantic** - Data validation and schemas
- **SQLite** - Metadata and ops storage
- **OpenAI API** - Semantic search embeddings (optional)
- **ClickHouse** - High-performance analytics serving layer (optional)
- **boto3** - S3 cloud data lake support (optional)

### Frontend
- **React 18** - UI framework
- **TypeScript** - Type-safe JavaScript
- **Vite** - Build tool
- **Tailwind CSS v4** - Styling
- **TanStack Query** - Data fetching and caching
- **Recharts** - Data visualization

## Installation

### Backend

```bash
# Clone the repository
git clone https://github.com/your-org/polymkt.git
cd polymkt

# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -e .

# Run the API server
uvicorn src.polymkt.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev     # Development server at http://localhost:5173
npm run build   # Production build
```

## Quick Start

### 1. Bootstrap Data

Place your CSV files in the `data/` directory:
- `data/markets.csv`
- `data/trades.csv`
- `data/orderFilled.csv`
- `data/events.csv` (optional, for tags)

```bash
# Run bootstrap import
curl -X POST http://localhost:8000/api/bootstrap
```

### 2. Build Search Index

```bash
# Build hybrid search index (BM25 + semantic)
curl -X POST http://localhost:8000/api/hybrid-search/build
```

### 3. Create a Dataset

Use the UI at `http://localhost:5173/datasets/new` or:

```bash
curl -X POST http://localhost:8000/api/datasets \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Election Markets",
    "description": "2024 US Election markets",
    "filters": {"query": "election", "category": "Politics"}
  }'
```

### 4. Run a Backtest

```bash
# Prepare backtest with natural language strategy
curl -X POST http://localhost:8000/api/backtest-agent/prepare \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "<dataset-id>",
    "natural_language_strategy": "buy favorite 90 days out, hold to expiry"
  }'

# Execute the prepared backtest
curl -X POST "http://localhost:8000/api/backtest-agent/execute?session_id=<session-id>"
```

## Architecture

```
polymkt/
├── src/polymkt/
│   ├── api/              # FastAPI endpoints
│   │   └── main.py       # ~4500 lines, 80+ endpoints
│   ├── pipeline/         # Data ingestion pipelines
│   │   ├── bootstrap.py  # CSV to Parquet conversion
│   │   ├── update.py     # Incremental updates
│   │   ├── curate.py     # Analytics layer creation
│   │   └── cloud_bootstrap.py  # S3 data lake
│   ├── storage/          # Data storage layers
│   │   ├── duckdb_layer.py     # DuckDB views
│   │   ├── parquet.py          # Parquet schemas and writers
│   │   ├── metadata.py         # SQLite metadata store
│   │   ├── search.py           # BM25 search index
│   │   ├── semantic_search.py  # OpenAI embeddings
│   │   ├── hybrid_search.py    # Combined search
│   │   ├── datasets.py         # Dataset persistence
│   │   ├── backtests.py        # Backtest persistence
│   │   ├── election_groups.py  # Election group management
│   │   └── clickhouse.py       # ClickHouse serving layer
│   ├── signals/          # Trading signals
│   │   └── favorites.py  # Favorite signal computation
│   ├── backtest/         # Backtest engine
│   │   └── engine.py     # Strategy execution
│   ├── agents/           # AI agents
│   │   ├── dataset_agent.py     # NL to dataset filters
│   │   └── backtesting_agent.py # NL to strategies
│   ├── services/         # Business services
│   │   ├── positions.py      # Wallet position tracking
│   │   └── llm_analytics.py  # Guarded LLM queries
│   ├── models/           # Pydantic schemas
│   │   └── schemas.py    # ~1600 lines of schemas
│   └── config.py         # Configuration
├── frontend/             # React frontend
│   ├── src/
│   │   ├── api/          # API client
│   │   ├── components/   # Shared components
│   │   ├── pages/        # Page components
│   │   └── types/        # TypeScript types
├── tests/                # Test suite (660 tests)
└── data/                 # Data directory
    ├── *.csv             # Source CSV files
    └── parquet/          # Parquet output
        ├── raw/          # Immutable raw layer
        └── analytics/    # Derived analytics layer
```

## Key Features

### Data Pipeline

| Feature | Description |
|---------|-------------|
| **Bootstrap Import** | Convert CSV to Parquet with ZSTD compression |
| **Field Normalization** | Normalize addresses, timestamps, numerics |
| **Partitioning** | Year/month/day + market_id hash bucket |
| **Raw/Analytics Layers** | Immutable raw layer, derived analytics layer |
| **Incremental Updates** | Watermark-based fetching, deduplication |
| **Schema Evolution** | Safe column access, mixed-schema partitions |

### Search

| Feature | Description |
|---------|-------------|
| **BM25 Full-Text** | DuckDB FTS with Porter stemmer |
| **Semantic Search** | OpenAI embeddings with HNSW index |
| **Hybrid Search** | RRF (Reciprocal Rank Fusion) merging |
| **Incremental Updates** | Content hash-based change detection |
| **Unified API** | Single endpoint with mode parameter |

### Backtesting

| Feature | Description |
|---------|-------------|
| **Election Groups** | Group related markets (candidates) |
| **Favorite Signals** | Highest YES price at N days to expiry |
| **Backtest Engine** | Entry/exit simulation with costs |
| **Fees & Slippage** | Configurable transaction costs |
| **Metrics** | Total return, win rate, Sharpe, drawdown |
| **Equity Curve** | Cumulative P&L visualization |

### Wallet Tracking (Sharp Money)

| Feature | Description |
|---------|-------------|
| **Static Watchlist** | Track known sharp traders |
| **Position Tracking** | Average-cost accounting per wallet |
| **Mark-to-Market** | 5-minute P&L snapshots |
| **Alerting** | Trade alerts with cooldown windows |
| **Metrics Rollups** | 1m/1h/1d aggregations |

### Infrastructure

| Feature | Description |
|---------|-------------|
| **ClickHouse** | Optional serving layer for rollups |
| **S3 Data Lake** | Optional cloud storage with verification |
| **Runtime Modes** | INGEST_MODE and ANALYTICS_MODE controls |
| **Data Quality** | Uniqueness, range, referential integrity checks |
| **Performance** | Sub-100ms single market, sub-500ms 100+ markets |

## API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/bootstrap` | POST | Run bootstrap import |
| `/api/update` | POST | Run incremental update |
| `/api/query/trades` | POST | Query trades with filters |
| `/api/markets/search` | GET | Unified market search |

### Datasets

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/datasets` | GET | List datasets |
| `/api/datasets` | POST | Create dataset |
| `/api/datasets/{id}` | GET | Get dataset |
| `/api/datasets/{id}` | PUT | Update dataset |
| `/api/datasets/{id}` | DELETE | Delete dataset |

### Backtests

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/backtests` | GET | List backtests |
| `/api/backtests` | POST | Create backtest |
| `/api/backtests/{id}` | GET | Get backtest |
| `/api/backtests/{id}/execute` | POST | Execute backtest |

### Agents

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/dataset-agent/query` | POST | NL to market list |
| `/api/backtest-agent/prepare` | POST | NL to strategy config |
| `/api/backtest-agent/execute` | POST | Execute prepared backtest |

### Sharp Money

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/sharp-money/watchlists` | GET/POST | Manage watchlists |
| `/api/sharp-money/alerts` | GET | List alerts |
| `/api/wallets/{address}/positions` | GET | Wallet positions |
| `/api/wallets/{address}/metrics` | GET | Wallet performance |

### Infrastructure

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/runtime/status` | GET | Runtime mode status |
| `/api/data-lake/status` | GET | S3 data lake status |
| `/api/clickhouse/status` | GET | ClickHouse status |
| `/api/llm-analytics/query/rollups` | POST | Guarded analytics query |

## Scheduled Data Pipelines

PolyMkt includes automated data pipelines that run on GitHub Actions to keep market data up to date in S3.

### Pipeline Overview

| Pipeline | Schedule | Description |
|----------|----------|-------------|
| **Gamma API Sync** | Hourly (`:00`) | Fetches markets and events from Polymarket API |
| **Trades Sync** | Every 10 min | Syncs order_filled events and derives trades |

### S3 Data Organization

All data is stored in `s3://polymarket-bcp892/`:

```
s3://polymarket-bcp892/
├── raw/polymarket/
│   ├── markets.parquet              # All markets (hourly refresh)
│   ├── events.parquet               # All events (hourly refresh)
│   ├── gamma_api/                   # Raw API audit trail
│   │   ├── markets/YYYY/MM/DD/      # Raw market JSONL by date
│   │   └── events/YYYY/MM/DD/       # Raw event JSONL by date
│   ├── order_filled/                # Order filled events
│   │   └── year=YYYY/month=MM/day=DD/
│   └── trades/                      # Derived trades
│       └── year=YYYY/month=MM/day=DD/
```

### Data Freshness SLAs

| Data | Freshness Target |
|------|------------------|
| Markets & Events | Within 1 hour |
| Order Filled | Within 10 minutes |
| Trades | Within 10 minutes |

### Manual Workflow Triggers

You can manually trigger workflows from the GitHub Actions UI:

1. Go to **Actions** → Select workflow (Gamma API Sync or Trades Sync)
2. Click **Run workflow**
3. Configure options:
   - **Gamma Sync**: Choose entity (all/markets/events), enable dry run
   - **Trades Sync**: Set max_batches for testing, enable dry run

Or use the GitHub CLI:

```bash
# Trigger Gamma API sync (dry run)
gh workflow run gamma-sync.yml -f entity=all -f dry_run=true

# Trigger Trades sync with limited batches
gh workflow run trades-sync.yml -f max_batches=10 -f dry_run=false
```

### Checking Pipeline Health

**1. View Recent Workflow Runs:**
```bash
# List recent runs
gh run list --workflow=gamma-sync.yml --limit=5
gh run list --workflow=trades-sync.yml --limit=5

# View details of a specific run
gh run view <run-id>
```

**2. Check S3 Data Freshness:**
```bash
# Check markets.parquet last modified
aws s3 ls s3://polymarket-bcp892/raw/polymarket/markets.parquet

# Check latest order_filled partitions
aws s3 ls s3://polymarket-bcp892/raw/polymarket/order_filled/ --recursive | tail -5

# Check latest trades partitions
aws s3 ls s3://polymarket-bcp892/raw/polymarket/trades/ --recursive | tail -5
```

**3. Verify Parquet Files:**
```bash
# Download and inspect markets
aws s3 cp s3://polymarket-bcp892/raw/polymarket/markets.parquet /tmp/
python -c "import pyarrow.parquet as pq; t = pq.read_table('/tmp/markets.parquet'); print(f'{t.num_rows} rows, columns: {t.column_names}')"
```

### GitHub Secrets Configuration

The workflows require these repository secrets:

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |

**To add or rotate secrets:**

1. Go to repository **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret** (or update existing)
3. Enter the secret name and value
4. Click **Add secret**

**Required IAM permissions for the user:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket",
        "s3:DeleteObject"
      ],
      "Resource": [
        "arn:aws:s3:::polymarket-bcp892",
        "arn:aws:s3:::polymarket-bcp892/*"
      ]
    }
  ]
}
```

### Troubleshooting

**Workflow failed with AWS credentials error:**
- Verify secrets are set correctly in repository settings
- Check IAM user has required S3 permissions
- Ensure access key is not expired or disabled

**Workflow timed out:**
- Gamma sync has 15-minute timeout, trades sync has 30-minute timeout
- For large catchups, run with `--max-batches` to limit data processed
- Check if API is rate-limiting (look for 429 errors in logs)

**Missing data in S3:**
- Verify workflow completed successfully (check Actions tab)
- Run workflow manually with `dry_run=false`
- Check S3 bucket permissions allow writes

**Partial data (some partitions missing):**
- Scripts are resumable - re-run the workflow
- Check logs for memory pressure warnings (scripts exit gracefully on high memory)
- For trades, ensure order_filled completed first (trades depend on it)

**High memory usage errors:**
- Scripts are designed to work with <2GB RAM
- Use `--max-batches` flag for testing locally
- Memory pressure triggers graceful exit at 80% system memory

### Running Scripts Locally

**Important:** Always use `--max-batches` when running locally to prevent memory exhaustion.

```bash
# Gamma API sync (dry run)
python scripts/s3_gamma_sync.py --dry-run

# Gamma API sync (limited batch for testing)
python scripts/s3_gamma_sync.py --entity markets

# Order filled sync (limited batches)
python scripts/s3_catchup.py --max-batches 10

# Trades derivation
python scripts/s3_trades_catchup.py
```

## Configuration

Environment variables (prefix: `POLYMKT_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `POLYMKT_DATA_DIR` | `data/` | Data directory |
| `POLYMKT_PARQUET_PARTITIONING_ENABLED` | `false` | Enable partitioning |
| `POLYMKT_PARQUET_HASH_BUCKET_COUNT` | `8` | Hash bucket count |
| `POLYMKT_OPENAI_API_KEY` | - | OpenAI API key (semantic search) |
| `POLYMKT_OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `POLYMKT_S3_ENABLED` | `false` | Enable S3 uploads |
| `POLYMKT_S3_BUCKET` | - | S3 bucket name |
| `POLYMKT_CLICKHOUSE_ENABLED` | `false` | Enable ClickHouse |
| `POLYMKT_CLICKHOUSE_HOST` | `localhost` | ClickHouse host |

## Testing

```bash
# Run backend tests
pytest

# Run frontend tests
cd frontend && npm test

# Type checking
mypy src/polymkt
```

**Test Coverage:**
- Backend: 660 tests
- Frontend: 38 tests
- Total: 698 tests

## Development Timeline

This project was developed over 8 days (2026-01-06 to 2026-01-13) with the following major milestones:

| Date | Features |
|------|----------|
| Jan 6 | Bootstrap, normalization, query interface, partitioning, layers |
| Jan 7 | Search (BM25, semantic, hybrid), datasets, backtests, agents |
| Jan 8 | Frontend setup, market search, pagination |
| Jan 11 | Strategy confirmation screen |
| Jan 12 | Backtest visualization, dataset editing, Sharp Money, runtime modes |
| Jan 13 | Wallet tracking, metrics rollups, S3, ClickHouse, LLM analytics |

## License

[Add your license here]

## Contributing

[Add contributing guidelines here]
