import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
from poly_utils.utils import get_markets, update_missing_tokens

import subprocess

import pandas as pd

def get_processed_df(df):
    markets_df = get_markets()
    markets_df = markets_df.rename({'id': 'market_id'})

    # 1) Make markets long: (market_id, side, asset_id) where side ∈ {"token1", "token2"}
    markets_long = (
        markets_df
        .select(["market_id", "token1", "token2"])
        .melt(id_vars="market_id", value_vars=["token1", "token2"],
            variable_name="side", value_name="asset_id")
    )

    # 2) Identify the non-USDC asset for each trade (the one that isn't 0)
    df = df.with_columns(
        pl.when(pl.col("makerAssetId") != "0")
        .then(pl.col("makerAssetId"))
        .otherwise(pl.col("takerAssetId"))
        .alias("nonusdc_asset_id")
    )

    # 3) Join once on that non-USDC asset to recover the market + side ("token1" or "token2")
    df = df.join(
        markets_long,
        left_on="nonusdc_asset_id",
        right_on="asset_id",
        how="left",
    )

    # 4) label columns and keep market_id
    df = df.with_columns([
        pl.when(pl.col("makerAssetId") == "0").then(pl.lit("USDC")).otherwise(pl.col("side")).alias("makerAsset"),
        pl.when(pl.col("takerAssetId") == "0").then(pl.lit("USDC")).otherwise(pl.col("side")).alias("takerAsset"),
        pl.col("market_id"),
    ])

    df = df[['timestamp', 'market_id', 'maker', 'makerAsset', 'makerAmountFilled', 'taker', 'takerAsset', 'takerAmountFilled', 'transactionHash']]

    df = df.with_columns([
        (pl.col("makerAmountFilled") / 10**6).alias("makerAmountFilled"),
        (pl.col("takerAmountFilled") / 10**6).alias("takerAmountFilled"),
    ])

    df = df.with_columns(
        pl.when(pl.col("takerAsset") == "USDC")
        .then(pl.lit("BUY"))
        .otherwise(pl.lit("SELL"))
        .alias("taker_direction")
    )

    df = df.with_columns([
        pl.when(pl.col("takerAsset") == "USDC")
        .then(pl.lit("BUY"))
        .otherwise(pl.lit("SELL"))
        .alias("taker_direction"),

        # reverse of taker_direction
        pl.when(pl.col("takerAsset") == "USDC")
        .then(pl.lit("SELL"))
        .otherwise(pl.lit("BUY"))
        .alias("maker_direction"),
    ])

    df = df.with_columns([
        pl.when(pl.col("makerAsset") != "USDC")
        .then(pl.col("makerAsset"))
        .otherwise(pl.col("takerAsset"))
        .alias("nonusdc_side"),

        pl.when(pl.col("takerAsset") == "USDC")
        .then(pl.col("takerAmountFilled"))
        .otherwise(pl.col("makerAmountFilled"))
        .alias("usd_amount"),
        pl.when(pl.col("takerAsset") != "USDC")
        .then(pl.col("takerAmountFilled"))
        .otherwise(pl.col("makerAmountFilled"))
        .alias("token_amount"),
        pl.when(pl.col("takerAsset") == "USDC")
        .then(pl.col("takerAmountFilled") / pl.col("makerAmountFilled"))
        .otherwise(pl.col("makerAmountFilled") / pl.col("takerAmountFilled"))
        .cast(pl.Float64)
        .alias("price")
    ])


    df = df[['timestamp', 'market_id', 'maker', 'taker', 'nonusdc_side', 'maker_direction', 'taker_direction', 'price', 'usd_amount', 'token_amount', 'transactionHash']]
    return df



def process_live():
    processed_file = 'processed/trades.csv'

    print("=" * 60)
    print("🔄 Processing Live Trades")
    print("=" * 60)

    last_processed = {}

    if os.path.exists(processed_file):
        print(f"✓ Found existing processed file: {processed_file}")
        result = subprocess.run(['tail', '-n', '1', processed_file], capture_output=True, text=True)
        last_line = result.stdout.strip()
        splitted = last_line.split(',')

        last_processed['timestamp'] = pd.to_datetime(splitted[0])
        last_processed['transactionHash'] = splitted[-1]
        last_processed['maker'] = splitted[2]
        last_processed['taker'] = splitted[3]
        
        print(f"📍 Resuming from: {last_processed['timestamp']}")
        print(f"   Last hash: {last_processed['transactionHash'][:16]}...")
    else:
        print("⚠ No existing processed file found - processing from beginning")

    print(f"\n📂 Reading: goldsky/orderFilled.csv")

    schema_overrides = {
        "takerAssetId": pl.Utf8,
        "makerAssetId": pl.Utf8,
    }

    df = pl.scan_csv("goldsky/orderFilled.csv", schema_overrides=schema_overrides).collect(streaming=True)
    df = df.with_columns(
        pl.from_epoch(pl.col('timestamp'), time_unit='s').alias('timestamp')
    )

    print(f"✓ Loaded {len(df):,} rows")

    df = df.with_row_index()

    same_timestamp = df.filter(pl.col('timestamp') == last_processed['timestamp'])
    same_timestamp = same_timestamp.filter(
        (pl.col("transactionHash") == last_processed['transactionHash']) & (pl.col("maker") == last_processed['maker']) & (pl.col("taker") == last_processed['taker'])
    )

    df_process = df.filter(pl.col('index') > same_timestamp.row(0)[0])
    df_process = df_process.drop('index')

    print(f"⚙️  Processing {len(df_process):,} new rows...")

    new_df = get_processed_df(df_process)
    
    if not os.path.isdir('processed'):
        os.makedirs('processed')


    op_file = 'processed/trades.csv'

    if not os.path.isfile(op_file):
        new_df.write_csv(op_file)
        print(f"✓ Created new file: processed/trades.csv")
    else:
        print(f"✓ Appending {len(new_df):,} rows to processed/trades.csv")
        with open(op_file, mode="a") as f:
            new_df.write_csv(f, include_header=False)

    
    print("=" * 60)
    print("✅ Processing complete!")
    print("=" * 60)
    

import os
import pandas as pd
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport
from flatten_json import flatten
from datetime import datetime, timezone
import subprocess
import time
from update_utils.update_markets import update_markets

# Global runtime timestamp - set once when program starts
RUNTIME_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# Columns to save
COLUMNS_TO_SAVE = ['timestamp', 'maker', 'makerAssetId', 'makerAmountFilled', 'taker', 'takerAssetId', 'takerAmountFilled', 'transactionHash']

if not os.path.isdir('goldsky'):
    os.mkdir('goldsky')

def get_latest_timestamp():
    """Get the latest timestamp from orderFilled.csv, or 0 if file doesn't exist"""
    cache_file = 'goldsky/orderFilled.csv'
    
    if not os.path.isfile(cache_file):
        print("No existing file found, starting from beginning of time (timestamp 0)")
        return 0
    
    try:
        # Use tail to get the last line efficiently
        result = subprocess.run(['tail', '-n', '1', cache_file], capture_output=True, text=True, check=True)
        last_line = result.stdout.strip()
        if last_line:
            # Get header to find timestamp column index
            header_result = subprocess.run(['head', '-n', '1', cache_file], capture_output=True, text=True, check=True)
            headers = header_result.stdout.strip().split(',')
            
            if 'timestamp' in headers:
                timestamp_index = headers.index('timestamp')
                values = last_line.split(',')
                if len(values) > timestamp_index:
                    last_timestamp = int(values[timestamp_index])
                    readable_time = datetime.fromtimestamp(last_timestamp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                    print(f'Resuming from timestamp {last_timestamp} ({readable_time})')
                    return last_timestamp
    except Exception as e:
        print(f"Error reading latest file with tail: {e}")
        # Fallback to pandas
        try:
            df = pd.read_csv(cache_file)
            if len(df) > 0 and 'timestamp' in df.columns:
                last_timestamp = df.iloc[-1]['timestamp']
                readable_time = datetime.fromtimestamp(int(last_timestamp), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                print(f'Resuming from timestamp {last_timestamp} ({readable_time})')
                return int(last_timestamp)
        except Exception as e2:
            print(f"Error reading with pandas: {e2}")
    
    # Fallback to beginning of time
    print("Falling back to beginning of time (timestamp 0)")
    return 0

def scrape(at_once=1000):
    QUERY_URL = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/0.0.1/gn"
    print(f"Query URL: {QUERY_URL}")
    print(f"Runtime timestamp: {RUNTIME_TIMESTAMP}")
    
    # Get starting timestamp from latest file
    last_value = get_latest_timestamp()
    count = 0
    total_records = 0

    print(f"\nStarting scrape for orderFilledEvents")
    
    output_file = 'goldsky/orderFilled.csv'
    print(f"Output file: {output_file}")
    print(f"Saving columns: {COLUMNS_TO_SAVE}")

    while True:
        q_string = '''query MyQuery {
                        orderFilledEvents(orderBy: timestamp 
                                             first: ''' + str(at_once) + '''
                                             where: {timestamp_gt: "''' + str(last_value) + '''"}) {
                            fee
                            id
                            maker
                            makerAmountFilled
                            makerAssetId
                            orderHash
                            taker
                            takerAmountFilled
                            takerAssetId
                            timestamp
                            transactionHash
                        }
                    }
                '''

        query = gql(q_string)
        transport = RequestsHTTPTransport(url=QUERY_URL, verify=True, retries=3)
        client = Client(transport=transport)
        
        try:
            res = client.execute(query)
        except Exception as e:
            print(f"Query error: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)
            continue
        
        if not res['orderFilledEvents'] or len(res['orderFilledEvents']) == 0:
            print(f"No more data for orderFilledEvents")
            break

        df = pd.DataFrame([flatten(x) for x in res['orderFilledEvents']]).reset_index(drop=True)
        
        # Sort by timestamp and update last_value
        df = df.sort_values('timestamp', ascending=True).reset_index(drop=True)
        last_value = df.iloc[-1]['timestamp']
        
        readable_time = datetime.fromtimestamp(int(last_value), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        print(f"Batch {count + 1}: Last timestamp {last_value} ({readable_time}), Records: {len(df)}")
        
        count += 1
        total_records += len(df)

        # Remove duplicates
        df = df.drop_duplicates()

        # Filter to only the columns we want to save
        df_to_save = df[COLUMNS_TO_SAVE].copy()

        # Save to file
        if os.path.isfile(output_file):
            df_to_save.to_csv(output_file, index=None, mode='a', header=None)
        else:
            df_to_save.to_csv(output_file, index=None)

        if len(df) < at_once:
            break

    print(f"Finished scraping orderFilledEvents")
    print(f"Total new records: {total_records}")
    print(f"Output file: {output_file}")

def update_goldsky():
    """Run scraping for orderFilledEvents"""
    print(f"\n{'='*50}")
    print(f"Starting to scrape orderFilledEvents")
    print(f"Runtime: {RUNTIME_TIMESTAMP}")
    print(f"{'='*50}")
    try:
        scrape()
        print(f"Successfully completed orderFilledEvents")
    except Exception as e:
        print(f"Error scraping orderFilledEvents: {str(e)}")

import requests
import csv
import json
import os
from typing import List, Dict

def count_csv_lines(csv_filename: str) -> int:
    """Count the number of data lines in CSV (excluding header)"""
    if not os.path.exists(csv_filename):
        return 0
    
    try:
        with open(csv_filename, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # Skip header
            return sum(1 for row in reader if row)  # Count non-empty rows
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return 0

def update_markets(csv_filename: str = "data/markets.csv", batch_size: int = 500):
    """
    Fetch markets ordered by creation date and save to CSV.
    Automatically resumes from the correct offset based on existing CSV lines.
    
    Args:
        csv_filename: Name of CSV file to save to
        batch_size: Number of markets to fetch per request
    """
    
    base_url = "https://gamma-api.polymarket.com/markets"
    
    # CSV headers for the required columns
    headers = [
        'createdAt', 'id', 'question', 'answer1', 'answer2', 'neg_risk', 
        'market_slug', 'token1', 'token2', 'condition_id', 'volume', 'ticker', 'closedTime',
        'description', 'category', 'tags'
    ]
    
    # Dynamically set offset based on existing records
    current_offset = count_csv_lines(csv_filename)
    file_exists = os.path.exists(csv_filename) and current_offset > 0
    
    if file_exists:
        print(f"Found {current_offset} existing records. Resuming from offset {current_offset}")
        mode = 'a'
    else:
        print(f"Creating new CSV file: {csv_filename}")
        mode = 'w'
    
    total_fetched = 0
    
    with open(csv_filename, mode, newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write headers only if file is new
        if mode == 'w':
            writer.writerow(headers)
        
        while True:
            print(f"Fetching batch at offset {current_offset}...")
            
            try:
                params = {
                    'order': 'createdAt',
                    'ascending': 'true',
                    'limit': batch_size,
                    'offset': current_offset
                }
                
                response = requests.get(base_url, params=params, timeout=30)
                
                # Handle different HTTP status codes
                if response.status_code == 500:
                    print(f"Server error (500) - retrying in 5 seconds...")
                    import time
                    time.sleep(5)
                    continue
                elif response.status_code == 429:
                    print(f"Rate limited (429) - waiting 10 seconds...")
                    import time
                    time.sleep(10)
                    continue
                elif response.status_code != 200:
                    print(f"API error {response.status_code}: {response.text}")
                    print("Retrying in 3 seconds...")
                    import time
                    time.sleep(3)
                    continue
                
                markets = response.json()
                
                if not markets:
                    print(f"No more markets found at offset {current_offset}. Completed!")
                    break
                
                batch_count = 0

                tags = []
                if market.get('tags') and len(market.get('tags', [])) > 0:
                    print(f"market.tags")
                    for d_tag in market['tags']:
                        if 'slug' in d_tag:
                            tags.append(d_tag.get('slug'))
                
                for market in markets:
                    try:
                        # Parse outcomes for answer1 and answer2
                        outcomes_str = market.get('outcomes', '[]')
                        if isinstance(outcomes_str, str):
                            outcomes = json.loads(outcomes_str)
                        else:
                            outcomes = outcomes_str
                        
                        answer1 = outcomes[0] if len(outcomes) > 0 else ''
                        answer2 = outcomes[1] if len(outcomes) > 1 else ''
                        
                        # Parse clobTokenIds for token1 and token2
                        clob_tokens_str = market.get('clobTokenIds', '[]')
                        if isinstance(clob_tokens_str, str):
                            clob_tokens = json.loads(clob_tokens_str)
                        else:
                            clob_tokens = clob_tokens_str
                        
                        token1 = clob_tokens[0] if len(clob_tokens) > 0 else ''
                        token2 = clob_tokens[1] if len(clob_tokens) > 1 else ''
                        
                        # Check for negative risk indicators
                        neg_risk = market.get('negRiskAugmented', False) or market.get('negRiskOther', False)
                        
                        # Create row with required columns
                        question_text = market.get('question', '') or market.get('title', '')

                        description = market.get('description', '')
                        category = market.get('category', '')
                        
                        # Get ticker from events if available
                        ticker = ''
                        if market.get('events') and len(market.get('events', [])) > 0:
                            ticker = market['events'][0].get('ticker', '')
                        
                        row = [
                            market.get('createdAt', ''),
                            market.get('id', ''),
                            question_text,
                            answer1,
                            answer2,
                            neg_risk,
                            market.get('slug', ''),
                            token1,
                            token2,
                            market.get('conditionId', ''),
                            market.get('volume', ''),
                            ticker,
                            market.get('closedTime', ''),
                            description,
                            category,
                            tags
                        ]
                        
                        writer.writerow(row)
                        batch_count += 1
                        
                    except (ValueError, KeyError, json.JSONDecodeError) as e:
                        print(f"Error processing market {market.get('id', 'unknown')}: {e}")
                        continue
                
                total_fetched += batch_count
                current_offset += batch_count  # Increment by actual records processed
                
                print(f"Processed {batch_count} markets. Total new: {total_fetched}. Next offset: {current_offset}")
                
                # Stop if we got fewer markets than expected (likely at the end)
                if len(markets) < batch_size:
                    print(f"Received only {len(markets)} markets (less than batch size). Reached end.")
                    break
                
            except requests.exceptions.RequestException as e:
                print(f"Network error: {e}")
                print(f"Retrying in 5 seconds...")
                import time
                time.sleep(5)
                continue
            except Exception as e:
                print(f"Unexpected error: {e}")
                print(f"Retrying in 3 seconds...")
                import time
                time.sleep(3)
                continue
    
    print(f"\nCompleted! Fetched {total_fetched} new markets.")
    print(f"Data saved to: {csv_filename}")
    print(f"Total records: {current_offset}")

if __name__ == "__main__":
    process_live()
    update_markets(batch_size=500)




    