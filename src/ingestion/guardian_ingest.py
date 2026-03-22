from __future__ import annotations
import os
import json
from pathlib import Path
from datetime import datetime, timezone
import requests

from src.utils.config import load_yaml
from src.storage.duckdb_client import DuckDBClient
from src.storage.parquet_io import write_parquet
from src.ingestion.normalize import normalize_records


def main():
    app_config = load_yaml("configs/config.yaml")
    sources_config = load_yaml("configs/sources.yaml")

    api_key = os.getenv(sources_config["guardian"]["api_key_env"])
    if not api_key:
        print("GUARDIAN_API_KEY not found. Skipping Guardian ingestion.")
        return

    url = "https://content.guardianapis.com/search"
    params = {
        "api-key": api_key,
        "section": sources_config["guardian"]["section"],
        "show-fields": "trailText,byline,headline",
        "page-size": 50,
        "order-by": "newest",
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    raw_dir = Path(app_config["paths"]["raw_dir"]) / "guardian"
    bronze_dir = Path(app_config["paths"]["bronze_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for item in payload.get("response", {}).get("results", []):
        fields = item.get("fields", {})
        records.append({
            "item_id": f"guardian_{item['id'].replace('/', '_')}",
            "source": "guardian",
            "source_type": "guardian",
            "title": fields.get("headline", item.get("webTitle", "")),
            "description": fields.get("trailText", ""),
            "full_text": None,
            "url": item.get("webUrl"),
            "author": fields.get("byline"),
            "published_at": item.get("webPublicationDate"),
            "fetched_at": datetime.now(timezone.utc),
            "category": item.get("sectionName", "news"),
            "topic": item.get("sectionName", "news"),
            "language": "en",
        })

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    with open(raw_dir / f"guardian_raw_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    df = normalize_records(records)
    write_parquet(df, bronze_dir / f"content_guardian_{timestamp}.parquet")

    db = DuckDBClient(app_config["paths"]["duckdb"])
    db.create_tables()
    db.insert_df("content_items", df)

    print(f"Ingested {len(df)} Guardian items.")


if __name__ == "__main__":
    main()