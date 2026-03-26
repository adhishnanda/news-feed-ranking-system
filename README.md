![Banner](docs/screenshots/banner.webp)

````markdown
# 🚀 Personalized News Feed Ranking System (V2)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![Model](https://img.shields.io/badge/Model-LightGBM-orange)
![Database](https://img.shields.io/badge/DB-DuckDB-yellow)
![Feature Store](https://img.shields.io/badge/FeatureStore-Redis-red)

---

## 🔥 One-Line Summary

A **production-inspired end-to-end recommender system** with retrieval, ranking, reranking, simulation, online feature store, and counterfactual evaluation.

---

## ⚡ Why This Project Is Different

Most student projects:
- train a model ❌  
- use static datasets ❌  
- stop at notebooks ❌  

This project:
- builds a **complete ML system loop** ✅  
- includes **real + simulated data** ✅  
- implements **online + offline architecture** ✅  
- demonstrates **ML + systems + product thinking** ✅  

---

## 🧠 Evolution

### V1 (MVP)
- Real ingestion (Hacker News + RSS)
- Event logging (impressions, clicks)
- Feature engineering
- Logistic Regression + LightGBM
- FastAPI + Streamlit
- Reranking (freshness + diversity)

---

### V2 (Production Upgrade 🚀)
- Simulation engine (30K interactions)
- Candidate retrieval layer
- Redis online feature store
- Advanced features (CTR, popularity, recency)
- Counterfactual evaluation (IPS / SNIPS)

---

## 🏗️ System Architecture

```mermaid
flowchart LR

A[Content Sources<br/>HN + RSS] --> B[Ingestion]
B --> C[(Storage<br/>DuckDB + Parquet)]

C --> D[Feature Engineering]
D --> E[Training Dataset]
E --> F[Model Training]
F --> G[Model Artifact]

C --> H[Simulation Engine]

G --> I[FastAPI API]

I --> J[Candidate Retrieval]
J --> K[Feature Assembly]
K --> L[Redis Feature Store]
L --> M[Ranking Model]
M --> N[Reranking Layer]
N --> O[Final Feed]

P[Streamlit UI] --> I
O --> P

P --> Q[User Events]
Q --> C
````

---

## 🧠 Core Components

### 1. Data Ingestion

* Hacker News API
* RSS feeds (BBC, TechCrunch, etc.)
* Unified schema
* Stored in DuckDB + Parquet

---

### 2. Simulation Engine 🚀

* Generates realistic user behavior
* Models:

  * click probability
  * position bias
  * recency bias

**Scale:**

* 100 users
* 300 items
* 30,000 interactions

---

### 3. Event Logging

Tracks:

* feed_request
* impression
* click
* save
* hide

Each event includes:

* user_id
* session_id
* item_id
* rank_position

---

### 4. Feature Engineering

#### User Features

* CTR
* interaction history
* preferences

#### Item Features

* CTR
* popularity
* freshness

#### Context Features

* time of day
* weekday

#### Interaction Features

* user-item match
* recency decay

---

### 5. Candidate Retrieval (Stage 1)

* popularity-based
* recency-based
* preference-based
* blended

---

### 6. Ranking (Stage 2)

* Logistic Regression
* LightGBM

Output:

* click probability

---

### 7. Reranking (Product Layer)

```
final_score = model_score + freshness_bonus - repetition_penalty
```

Optimizes:

* freshness
* diversity
* source balance

---

### 8. Redis Feature Store 🚀

* Online feature serving
* user + item features
* real-time inference

---

### 9. API Serving

Endpoint:

```
POST /rank-feed-v2
```

Pipeline:
retrieve → features → Redis → score → rerank → return

---

### 10. Evaluation

#### Ranking Metrics

* Precision@K
* Recall@K
* NDCG@K

#### Counterfactual Evaluation

* IPS
* SNIPS

---

## 📊 Metrics Snapshot

**Dataset**

* 30,000 interactions
* 100 users
* 300 items

**Ranking**

* Precision@5 ≈ 0.34
* Recall@5 ≈ 0.53
* NDCG@5 ≈ 0.83

**Counterfactual**

* IPS ≈ 0.2604
* SNIPS ≈ 0.2585

---

## 🔁 End-to-End Loop

User → Feed → Interaction → Events → Features → Training → Model → API → Better Feed

---

## 🎯 What This Demonstrates

### ML

* ranking systems
* implicit feedback
* feature engineering

### Systems

* ingestion pipelines
* API serving
* feature store

### Product

* diversity vs relevance
* freshness vs engagement

---

## ⚠️ Limitations

* simulated data
* no embeddings yet
* no A/B testing

---

## 🚀 Future Work

* embeddings-based personalization
* deep learning ranking
* exploration strategies

---

## 📸 Screenshots

### Streamlit Feed

### API Response

### Architecture

### Metrics

### Redis

---

## 🧠 Final Takeaway

Not a notebook.
Not a toy project.

👉 A **production-inspired recommender system**.

---

```
```

