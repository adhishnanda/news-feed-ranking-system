from pathlib import Path
import duckdb
import pandas as pd


class DuckDBClient:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path

    def connect(self):
        return duckdb.connect(self.db_path)

    def create_tables(self) -> None:
        with self.connect() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS content_items (
                    item_id VARCHAR,
                    source VARCHAR,
                    source_type VARCHAR,
                    title VARCHAR,
                    description VARCHAR,
                    full_text VARCHAR,
                    url VARCHAR,
                    author VARCHAR,
                    published_at TIMESTAMP,
                    fetched_at TIMESTAMP,
                    category VARCHAR,
                    topic VARCHAR,
                    language VARCHAR,
                    content_length INTEGER
                )
            """)

            con.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id VARCHAR,
                    timestamp TIMESTAMP,
                    user_id VARCHAR,
                    session_id VARCHAR,
                    event_type VARCHAR,
                    item_id VARCHAR,
                    rank_position INTEGER,
                    model_version VARCHAR,
                    score DOUBLE,
                    policy_name VARCHAR,
                    propensity DOUBLE,
                    dwell_time DOUBLE,
                    device_type VARCHAR,
                    metadata VARCHAR
                )
            """)

    def insert_df(self, table_name: str, df: pd.DataFrame) -> None:
        with self.connect() as con:
            con.register("tmp_df", df)
            con.execute(f"INSERT INTO {table_name} SELECT * FROM tmp_df")

    def query_df(self, sql: str) -> pd.DataFrame:
        with self.connect() as con:
            return con.execute(sql).fetchdf()