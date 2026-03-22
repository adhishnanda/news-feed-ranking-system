PERSONALIZED NEWS FEED RANKING SYSTEM
====================================

1. PROJECT OVERVIEW
-------------------

This project implements a production-inspired personalized news feed ranking system.

It simulates how modern platforms (news apps, social feeds) rank content using:

- real content ingestion
- event logging
- feature engineering
- machine learning ranking
- multi-objective reranking
- API-based serving
- interactive UI

The goal is to build a realistic, end-to-end ML system — not just a model.


2. WHY THIS PROJECT
-------------------

Most student projects are limited to:
- dashboards
- simple classification models

This project demonstrates:

- real-world ML system design
- ranking systems (not just classification)
- event-driven data pipelines
- feature engineering (user, item, context)
- offline vs online ML thinking
- product-aware ML decisions
- end-to-end architecture

This makes it highly valuable for:
- Data Science roles
- Machine Learning Engineering roles
- Recommender Systems roles
- Data Engineering / ML Systems roles


3. SYSTEM ARCHITECTURE
----------------------

High-level flow:

Streamlit UI
    ↓
FastAPI Ranking Service
    ↓
Feature Layer (offline features)
    ↓
ML Model (Logistic Regression)
    ↓
Reranking Layer (freshness + diversity)
    ↓
Ranked Feed → UI
    ↓
User Interactions → Event Logging → DuckDB


4. TECH STACK
-------------

Core:
- Python 3.11
- DuckDB (analytics DB)
- Parquet (storage)
- pandas

ML:
- scikit-learn (Logistic Regression)

Serving:
- FastAPI
- Uvicorn

UI:
- Streamlit

Other:
- requests
- feedparser

Planned:
- LightGBM
- Redis
- sentence-transformers


5. DATA INGESTION
-----------------

Sources:
- Hacker News API
- RSS feeds (BBC, TechCrunch, VentureBeat)

Steps:
- fetch raw data
- normalize schema
- deduplicate
- store in DuckDB

Output:
content_items table


6. EVENT LOGGING
----------------

Events tracked:

- feed_request
- impression
- click
- save
- hide

Each event contains:
- user_id
- session_id
- item_id
- timestamp
- rank_position
- model_score (optional)

Why this matters:
- builds ranking dataset
- enables implicit feedback learning


7. FEATURE ENGINEERING
----------------------

Item Features:
- age_hours
- title_length
- description_length
- source
- category
- publish hour/day

User Features:
- recent_impression_count
- recent_click_count
- recent_save_count
- recent_hide_count
- preferred_source
- preferred_category

Context Features:
- hour_of_day
- weekday
- is_weekend

All features stored in Parquet (offline feature store).


8. TRAINING DATASET
-------------------

Built from:

Impressions → all shown items  
Clicks → positive signals  

Target:
clicked = 1 if user clicked, else 0

This creates a ranking dataset suitable for:
- pointwise ranking models


9. MODELING
-----------

Baseline Model:
- Logistic Regression

Pipeline:
- numeric + categorical features
- imputation
- one-hot encoding

Output:
- probability of click (model_score)

Note:
Dataset is small → results are for pipeline validation.


10. RANKING PIPELINE
--------------------

Steps:

1. Candidate generation (recent items)
2. Feature construction (user + item + context)
3. Model scoring
4. Multi-objective reranking
5. Return ranked items


11. MULTI-OBJECTIVE RERANKING
-----------------------------

Final score:

final_score = model_score + freshness_bonus - diversity_penalty - source_penalty

Components:

Freshness:
- newer items get higher score

Diversity:
- penalize repeated categories

Source repetition:
- penalize repeated sources

Why this matters:
- prevents repetitive feeds
- improves user experience


12. API DESIGN
--------------

Endpoints:

GET /health  
→ health check

POST /rank-feed  
→ returns ranked feed

Input:
- user_id
- session_id
- limit

Output:
- ranked items
- model_score
- freshness_bonus
- final_rank


13. UI (STREAMLIT)
------------------

Features:

- user selection
- session tracking
- ranked feed display
- model score visualization
- interaction buttons:
  - save
  - hide
  - click

The UI calls the FastAPI backend.


14. HOW TO RUN
--------------

Step 1 — install dependencies

pip install -r requirements.txt


Step 2 — run ingestion

python -m src.ingestion.hn_ingest
python -m src.ingestion.rss_ingest


Step 3 — build features

python -m src.features.item_features
python -m src.features.user_features
python -m src.features.context_features
python -m src.features.dataset_builder


Step 4 — train model

python -m src.models.train


Step 5 — start API

uvicorn src.api.main:app --reload --port 8000


Step 6 — start UI

streamlit run src/ui/streamlit_app.py


15. KEY DESIGN DECISIONS
------------------------

Why DuckDB:
- lightweight
- fast
- no infrastructure needed

Why Logistic Regression:
- simple baseline
- interpretable

Why reranking layer:
- CTR-only ranking is not enough
- improves diversity and freshness

Why Streamlit + FastAPI:
- separates UI and serving
- mimics real-world architecture


16. LIMITATIONS
---------------

- small dataset
- no real users at scale
- no online A/B testing
- no strict point-in-time features
- simple model


17. FUTURE IMPROVEMENTS
-----------------------

- LightGBM / XGBoost
- embeddings (sentence-transformers)
- Redis online feature store
- counterfactual evaluation (IPS, DR)
- bandit-based ranking
- larger dataset / simulator


18. RESUME BULLET
-----------------

Built a production-inspired personalized news feed ranking system using Python, DuckDB, FastAPI, and Streamlit with real-time content ingestion, event logging, feature engineering, ML-based ranking, and multi-objective reranking.


19. KEY LEARNINGS
-----------------

- ranking systems differ from classification
- importance of impression vs click logging
- feature consistency is critical
- product tradeoffs matter
- simplicity beats overengineering
