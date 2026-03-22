# Architecture

The system follows a production-inspired design:

## Components

### UI
Streamlit frontend for user interaction.

### API
FastAPI service for ranking requests.

### Feature Layer
Offline feature store built using Parquet.

### Model
Logistic regression baseline.

### Reranking
Adds diversity and freshness constraints.

### Storage
DuckDB used for analytics and event storage.

---

## Flow

1. User requests feed
2. API generates candidates
3. Features are retrieved
4. Model scores items
5. Reranking adjusts scores
6. Feed returned
7. Events logged