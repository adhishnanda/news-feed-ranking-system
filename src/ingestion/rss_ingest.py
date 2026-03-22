from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json
import feedparser

from src.utils.config import load_yaml
from src.storage.duckdb_client import DuckDBClient
from src.storage.parquet_io import write_parquet
from src.ingestion.normalize import normalize_records, make_rss_item_id


def parse_entry(feed_name: str, category: str, entry) -> dict:
    published = None
    if getattr(entry, "published_parsed", None):
        published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)

    title = getattr(entry, "title", "") or ""
    description = getattr(entry, "summary", "") or ""
    url = getattr(entry, "link", "") or ""

    return {
        "item_id": make_rss_item_id(url, title),
        "source": feed_name,
        "source_type": "rss",
        "title": title,
        "description": description,
        "full_text": None,
        "url": url,
        "author": getattr(entry, "author", None),
        "published_at": published,
        "fetched_at": datetime.now(timezone.utc),
        "category": category,
        "topic": category,
        "language": "en",
    }


def main():
    app_config = load_yaml("configs/config.yaml")
    sources_config = load_yaml("configs/sources.yaml")

    raw_dir = Path(app_config["paths"]["raw_dir"]) / "rss"
    bronze_dir = Path(app_config["paths"]["bronze_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    parsed_raw = {}
    records = []

    for feed in sources_config["rss_feeds"]:
        try:
            feed_obj = feedparser.parse(feed["url"])
            parsed_raw[feed["name"]] = {
                "feed": dict(feed_obj.feed),
                "entries_count": len(feed_obj.entries),
            }

            for entry in feed_obj.entries[:50]:
                records.append(parse_entry(feed["name"], feed["category"], entry))
        except Exception as e:
            print(f"Failed RSS feed {feed['name']}: {e}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    with open(raw_dir / f"rss_raw_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(parsed_raw, f, ensure_ascii=False, indent=2, default=str)

    df = normalize_records(records)
    write_parquet(df, bronze_dir / f"content_rss_{timestamp}.parquet")

    db = DuckDBClient(app_config["paths"]["duckdb"])
    db.create_tables()
    db.insert_df("content_items", df)

    print(f"Ingested {len(df)} RSS items.")


if __name__ == "__main__":
    main()