# System Architecture

## High-Level Design

The system follows a modern production recommender architecture:

1. Data ingestion
2. Event logging
3. Feature engineering
4. Offline training
5. Online feature store
6. Retrieval
7. Ranking
8. Reranking
9. API serving

---

## Components

### 1. Ingestion Layer
- Collects news data (RSS / HackerNews)
- Stores raw data in bronze/silver layers

### 2. Event Logging
- Tracks:
  - impressions
  - clicks
  - saves
  - hides
- Stored as structured logs

---

### 3. Feature Engineering

#### User Features
- historical clicks
- CTR
- preferences

#### Item Features
- CTR
- popularity
- age
- metadata

#### Context Features
- time of day
- weekday
- session features

---

### 4. Offline Training

- Uses time-based splits
- Avoids data leakage
- Produces ranking model

---

### 5. Online Feature Store (Redis)

Purpose:
- Serve latest features at inference time

Stored:
- user features
- item features

---

### 6. Candidate Retrieval

Multiple strategies:
- popularity-based
- recency-based
- preference-based

---

### 7. Ranking Model

- Logistic Regression (baseline)
- LightGBM (optional)

---

### 8. Reranking Layer

Optimizes:
- diversity
- freshness
- relevance

---

### 9. API Layer

- FastAPI
- End-to-end pipeline:
  - retrieve → rank → rerank → return

---