"""Application configuration."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_prefix="POLYMKT_", env_file=".env", extra="ignore")

    # Data paths
    data_dir: Path = Path("data")
    parquet_dir: Path = Path("data/parquet")
    parquet_raw_dir: Path = Path("data/parquet/raw")
    parquet_analytics_dir: Path = Path("data/parquet/analytics")
    duckdb_path: Path = Path("data/polymkt.duckdb")
    metadata_db_path: Path = Path("data/metadata.db")

    # CSV file paths (for bootstrap)
    markets_csv: Path = Path("data/markets.csv")
    trades_csv: Path = Path("data/trades.csv")
    order_filled_csv: Path = Path("data/orderFilled.csv")
    events_csv: Path = Path("data/events.csv")

    # Parquet settings
    parquet_compression: str = "zstd"
    parquet_row_group_size: int = 100_000

    # Partitioning settings
    parquet_partitioning_enabled: bool = False  # Set True to enable year/month/day/hash_bucket partitioning
    parquet_hash_bucket_count: int = 8  # Number of hash buckets for market_id

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # OpenAI settings for semantic search
    openai_api_key: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int = 1536

    # S3 settings for cloud data lake
    s3_bucket: str | None = None
    s3_region: str = "us-east-1"
    s3_raw_prefix: str = "raw/polymarket"
    s3_curated_prefix: str = "curated/polymarket"
    s3_enabled: bool = False  # Set True to enable S3 upload

    # ClickHouse settings for analytics serving layer
    clickhouse_host: str = "localhost"
    clickhouse_port: int = 9000
    clickhouse_http_port: int = 8123
    clickhouse_database: str = "polymkt"
    clickhouse_user: str = "default"
    clickhouse_password: str = ""
    clickhouse_enabled: bool = False  # Set True to enable ClickHouse
    clickhouse_raw_retention_days: int = 30  # Days to keep raw events


settings = Settings()
