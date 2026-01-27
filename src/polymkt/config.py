"""Application configuration."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class BootstrapConfig(BaseModel):
    """Configuration for memory-bounded bootstrap processing.

    Allows tuning batch sizes and memory limits to prevent crashes
    on machines with limited resources processing large datasets.
    """

    # Batch processing - reduced from 100K default to 10K for safety
    batch_size: int = 10_000

    # Memory limits - process will save checkpoint and stop if exceeded
    max_memory_mb: int = 4096  # Default 4GB limit
    memory_check_interval: int = 10  # Check memory every N batches

    # Checkpointing for resume capability
    checkpoint_enabled: bool = True
    checkpoint_interval: int = 100  # Save checkpoint every N batches
    checkpoint_dir: Path = Path("data/.bootstrap_checkpoints")

    # DuckDB settings for joins
    duckdb_memory_limit_mb: int = 1024  # 1GB for joins

    # Events handling - spill to disk if exceeds this
    events_max_in_memory: int = 100_000

    def model_post_init(self, __context: Any) -> None:
        """Ensure checkpoint directory exists."""
        if self.checkpoint_enabled:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


class RewriteConfig(BaseModel):
    """Configuration for memory-efficient S3 parquet rewrite operations.

    Allows rewriting large parquet files on S3 without loading entire
    datasets into memory. Processes file-by-file, streaming batches.
    """

    # S3 settings
    s3_bucket: str = "polymarket-bcp892"
    s3_prefix: str = "raw/polymarket"
    staging_prefix: str = "staging/polymarket"
    s3_region: str = "us-east-1"

    # Batch processing
    batch_size: int = 10_000

    # Memory limits - leave headroom below actual limit
    max_memory_mb: int = 6000  # 8GB limit - 2GB headroom
    memory_check_interval: int = 10  # Check memory every N batches

    # Checkpointing for resume capability
    checkpoint_enabled: bool = True
    checkpoint_interval: int = 50  # Save checkpoint every N batches
    checkpoint_dir: Path = Path("data/.rewrite_checkpoints")

    # Verification
    verify_row_counts: bool = True

    # Cleanup
    delete_staging_on_success: bool = True

    def model_post_init(self, __context: Any) -> None:
        """Ensure checkpoint directory exists."""
        if self.checkpoint_enabled:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


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
    s3_streaming: bool = False  # Set True to stream directly to S3 (bypasses local disk)

    # ClickHouse settings for analytics serving layer
    clickhouse_host: str = "localhost"
    clickhouse_port: int = 9000
    clickhouse_http_port: int = 8123
    clickhouse_database: str = "polymkt"
    clickhouse_user: str = "default"
    clickhouse_password: str = ""
    clickhouse_enabled: bool = False  # Set True to enable ClickHouse
    clickhouse_raw_retention_days: int = 30  # Days to keep raw events

    # Bootstrap settings for memory-bounded processing
    bootstrap_batch_size: int = 10_000  # Rows per batch (reduced from 100K)
    bootstrap_max_memory_mb: int = 4096  # Memory limit in MB
    bootstrap_checkpoint_enabled: bool = True  # Enable checkpoint/resume
    bootstrap_checkpoint_interval: int = 100  # Checkpoint every N batches
    bootstrap_events_max_in_memory: int = 100_000  # Spill events to disk if larger

    # Rewrite settings for S3 parquet rewrites
    rewrite_batch_size: int = 10_000  # Rows per batch
    rewrite_max_memory_mb: int = 6000  # Memory limit (8GB - 2GB headroom)
    rewrite_checkpoint_enabled: bool = True  # Enable checkpoint/resume
    rewrite_checkpoint_interval: int = 50  # Checkpoint every N batches
    rewrite_staging_prefix: str = "staging/polymarket"  # Staging location on S3

    def get_bootstrap_config(self) -> BootstrapConfig:
        """Create BootstrapConfig from environment settings."""
        return BootstrapConfig(
            batch_size=self.bootstrap_batch_size,
            max_memory_mb=self.bootstrap_max_memory_mb,
            checkpoint_enabled=self.bootstrap_checkpoint_enabled,
            checkpoint_interval=self.bootstrap_checkpoint_interval,
            events_max_in_memory=self.bootstrap_events_max_in_memory,
        )

    def get_rewrite_config(self) -> RewriteConfig:
        """Create RewriteConfig from environment settings."""
        return RewriteConfig(
            s3_bucket=self.s3_bucket or "polymarket-bcp892",
            s3_prefix=self.s3_raw_prefix,
            staging_prefix=self.rewrite_staging_prefix,
            s3_region=self.s3_region,
            batch_size=self.rewrite_batch_size,
            max_memory_mb=self.rewrite_max_memory_mb,
            checkpoint_enabled=self.rewrite_checkpoint_enabled,
            checkpoint_interval=self.rewrite_checkpoint_interval,
        )


settings = Settings()
