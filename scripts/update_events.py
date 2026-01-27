"""Fetch events from Polymarket gamma API and save to CSV.

This script follows the same pattern as poly_data's update_markets.py,
fetching events with pagination and saving to events.csv for bootstrap ingestion.
"""

import csv
import json
import os
import time
from typing import Any

import requests


def count_csv_lines(file_path: str) -> int:
    """Count lines in CSV file (excluding header) for resume capability."""
    if not os.path.exists(file_path):
        return 0
    with open(file_path, "r") as f:
        return sum(1 for _ in f) - 1  # Subtract header


def update_events(
    output_path: str = "data/events.csv",
    batch_size: int = 500,
    max_retries: int = 5,
) -> int:
    """
    Fetch events from Polymarket gamma API and save to CSV.

    Args:
        output_path: Path to output CSV file
        batch_size: Number of records per API request
        max_retries: Maximum retries on failure

    Returns:
        Total number of events fetched
    """
    api_url = "https://gamma-api.polymarket.com/events"

    # Calculate starting offset from existing data
    existing_count = count_csv_lines(output_path)
    offset = existing_count
    total_fetched = existing_count

    print(f"Starting from offset {offset} (existing records: {existing_count})")

    # Determine if we need to write headers
    write_header = not os.path.exists(output_path) or existing_count == 0

    # Open file in append mode
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow(["eventId", "tags", "title", "description", "createdAt"])

        while True:
            params = {
                "limit": batch_size,
                "offset": offset,
                "order": "createdAt",
                "ascending": "true",
            }

            retries = 0
            response = None

            while retries < max_retries:
                try:
                    response = requests.get(api_url, params=params, timeout=30)

                    if response.status_code == 200:
                        break
                    elif response.status_code == 500:
                        print(f"Server error 500, retrying in 5s... (attempt {retries + 1})")
                        time.sleep(5)
                        retries += 1
                    elif response.status_code == 429:
                        print(f"Rate limited, waiting 10s... (attempt {retries + 1})")
                        time.sleep(10)
                        retries += 1
                    else:
                        print(f"HTTP {response.status_code}: {response.text[:200]}")
                        retries += 1
                        time.sleep(2)

                except requests.exceptions.RequestException as e:
                    print(f"Request failed: {e}, retrying in 5s...")
                    time.sleep(5)
                    retries += 1

            if response is None or response.status_code != 200:
                print(f"Failed after {max_retries} retries, stopping.")
                break

            events = response.json()

            if not events:
                print("No more events returned, finished.")
                break

            for event in events:
                # Extract event fields
                event_id = event.get("id", "")
                title = event.get("title", "")
                description = event.get("description", "")
                created_at = event.get("createdAt", "")

                # Handle tags - extract just the label from each tag object
                tags_raw = event.get("tags", [])
                if tags_raw is None:
                    tags_raw = []
                elif isinstance(tags_raw, str):
                    try:
                        tags_raw = json.loads(tags_raw)
                    except json.JSONDecodeError:
                        tags_raw = [tags_raw] if tags_raw else []

                # Extract tag labels from tag objects (API returns [{id, label, slug, ...}])
                tag_labels = []
                for tag in tags_raw:
                    if isinstance(tag, dict):
                        label = tag.get("label") or tag.get("slug") or tag.get("id", "")
                        if label:
                            tag_labels.append(str(label))
                    elif isinstance(tag, str):
                        tag_labels.append(tag)

                # Convert tags to JSON string for CSV storage
                tags_json = json.dumps(tag_labels)

                writer.writerow([event_id, tags_json, title, description, created_at])
                total_fetched += 1

            # Flush to disk periodically
            f.flush()

            print(f"Fetched {len(events)} events (total: {total_fetched}, offset: {offset})")

            # Check if we've reached the end
            if len(events) < batch_size:
                print("Received fewer events than batch size, finished.")
                break

            offset += batch_size

            # Small delay to be nice to the API
            time.sleep(0.5)

    print(f"Done! Total events: {total_fetched}")
    return total_fetched


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch Polymarket events to CSV")
    parser.add_argument(
        "--output", "-o",
        default="data/events.csv",
        help="Output CSV path (default: data/events.csv)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=500,
        help="Batch size per request (default: 500)"
    )
    args = parser.parse_args()

    update_events(output_path=args.output, batch_size=args.batch_size)
