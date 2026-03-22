```mermaid
flowchart TD

UI[Streamlit UI]
API[FastAPI Ranking API]
MODEL[ML Model]
RERANK[Reranking Layer]
FEATURES[Feature Store (Parquet)]
DB[(DuckDB)]
EVENTS[(Events)]

UI --> API
API --> FEATURES
API --> MODEL
MODEL --> RERANK
RERANK --> API
API --> UI

UI --> EVENTS
EVENTS --> DB
FEATURES --> DB
DB --> FEATURES