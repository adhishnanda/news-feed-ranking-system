from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import pandas as pd


def now_utc():
    return datetime.now(timezone.utc)


def make_rss_item_id(url: str, title: str) -> str:
    base = f"{url}|{title}".encode("utf-8", errors="ignore")
    return "rss_" + hashlib.md5(base).hexdigest()


def normalize_records(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)

    expected = [
        "item_id", "source", "source_type", "title", "description", "full_text",
        "url", "author", "published_at", "fetched_at", "category", "topic",
        "language", "content_length"
    ]

    for col in expected:
        if col not in df.columns:
            df[col] = None

    df["title"] = df["title"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df["content_length"] = (df["title"] + " " + df["description"]).str.len()

    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True, errors="coerce")
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")

    df = df.drop_duplicates(subset=["url"], keep="first")
    df = df.drop_duplicates(subset=["item_id"], keep="first")

    return df[expected]