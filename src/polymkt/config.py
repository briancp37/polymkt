"""Application configuration."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_prefix="POLYMKT_", env_file=".env", extra="ignore")

    # Data paths
    data_dir: Path = Path("data")
    parquet_dir: Path = Path("data/parquet")
    duckdb_path: Path = Path("data/polymkt.duckdb")
    metadata_db_path: Path = Path("data/metadata.db")

    # CSV file paths (for bootstrap)
    markets_csv: Path = Path("data/markets.csv")
    trades_csv: Path = Path("data/trades.csv")
    order_filled_csv: Path = Path("data/orderFilled.csv")

    # Parquet settings
    parquet_compression: str = "zstd"
    parquet_row_group_size: int = 100_000

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000


settings = Settings()
