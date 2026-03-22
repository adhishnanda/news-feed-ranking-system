from src.utils.config import load_yaml
from src.storage.duckdb_client import DuckDBClient

cfg = load_yaml("configs/config.yaml")
db = DuckDBClient(cfg["paths"]["duckdb"])

print("=== Event counts by type ===")
df_counts = db.query_df("""
    SELECT event_type, COUNT(*) AS n
    FROM events
    GROUP BY 1
    ORDER BY n DESC
""")
print(df_counts)

print("\n=== Event counts by user ===")
df_users = db.query_df("""
    SELECT user_id, event_type, COUNT(*) AS n
    FROM events
    GROUP BY 1, 2
    ORDER BY user_id, n DESC
""")
print(df_users)

print("\n=== Sample events ===")
df_sample = db.query_df("""
    SELECT timestamp, user_id, session_id, event_type, item_id, rank_position
    FROM events
    ORDER BY timestamp DESC
    LIMIT 20
""")
print(df_sample)