from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
import requests

from src.utils.config import load_yaml
from src.storage.duckdb_client import DuckDBClient
from src.storage.parquet_io import write_parquet
from src.ingestion.normalize import normalize_records


def fetch_hn_topstories(config: dict) -> list[int]:
    url = config["hackernews"]["topstories_endpoint"]
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    ids = resp.json()
    return ids[: config["hackernews"]["max_items"]]


def fetch_hn_item(item_id: int, template: str) -> dict:
    url = template.format(item_id=item_id)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def normalize_hn_item(item: dict) -> dict | None:
    if not item or item.get("type") != "story":
        return None
    if not item.get("url"):
        return None

    published_at = None
    if item.get("time"):
        published_at = datetime.fromtimestamp(item["time"], tz=timezone.utc)

    return {
    "item_id": f"hn_{item['id']}",
    "source": "hackernews",
    "source_type": "hackernews",
    "title": item.get("title", ""),
    "description": "",
    "full_text": None,
    "url": item.get("url"),
    "author": item.get("by"),
    "published_at": published_at,
    "fetched_at": datetime.now(timezone.utc),
    "category": "tech",
    "topic": "tech",
    "language": "en",
    }
    
def main():
    app_config = load_yaml("configs/config.yaml")
    sources_config = load_yaml("configs/sources.yaml")

    raw_dir = Path(app_config["paths"]["raw_dir"]) / "hackernews"
    bronze_dir = Path(app_config["paths"]["bronze_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    ids = fetch_hn_topstories(sources_config)
    template = sources_config["hackernews"]["item_endpoint_template"]

    raw_items = []
    normalized = []

    for item_id in ids:
        try:
            item = fetch_hn_item(item_id, template)
            raw_items.append(item)
            norm = normalize_hn_item(item)
            if norm:
                normalized.append(norm)
        except Exception as e:
            print(f"Failed HN item {item_id}: {e}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    with open(raw_dir / f"hn_raw_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(raw_items, f, ensure_ascii=False, indent=2, default=str)

    df = normalize_records(normalized)
    write_parquet(df, bronze_dir / f"content_hn_{timestamp}.parquet")

    db = DuckDBClient(app_config["paths"]["duckdb"])
    db.create_tables()
    db.insert_df("content_items", df)

    print(f"Ingested {len(df)} Hacker News items.")


if __name__ == "__main__":
    main()