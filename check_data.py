from src.utils.config import load_yaml
from src.storage.duckdb_client import DuckDBClient

cfg = load_yaml("configs/config.yaml")
db = DuckDBClient(cfg["paths"]["duckdb"])

print("=== Counts by source_type and source ===")
df_counts = db.query_df("""
    SELECT source_type, source, COUNT(*) AS n
    FROM content_items
    GROUP BY 1, 2
    ORDER BY n DESC
""")
print(df_counts)

print("\n=== Sample rows ===")
df_sample = db.query_df("""
    SELECT item_id, title, source, source_type, author, published_at, url
    FROM content_items
    LIMIT 10
""")
print(df_sample)

print("\n=== Null/empty URL check ===")
df_nulls = db.query_df("""
    SELECT COUNT(*) AS bad_urls
    FROM content_items
    WHERE url IS NULL OR TRIM(url) = ''
""")
print(df_nulls)

print("\n=== Duplicate URL check ===")
df_dupes = db.query_df("""
    SELECT COUNT(*) AS duplicate_url_groups
    FROM (
        SELECT url
        FROM content_items
        GROUP BY url
        HAVING COUNT(*) > 1
    )
""")
print(df_dupes)

print("\n=== Null/empty title check ===")
df_bad_titles = db.query_df("""
    SELECT COUNT(*) AS bad_titles
    FROM content_items
    WHERE title IS NULL OR TRIM(title) = ''
""")
print(df_bad_titles)

print("\n=== Total rows ===")
df_total = db.query_df("""
    SELECT COUNT(*) AS total_rows
    FROM content_items
""")
print(df_total)