install:
	pip install -r requirements.txt

run-redis:
	docker compose up -d redis

run-api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	streamlit run src/ui/streamlit_app.py

ingest:
	python -m src.ingestion.hn_ingest
	python -m src.ingestion.rss_ingest

test:
	pytest tests/ -v

demo:
	@echo "Starting Redis (background)..."
	docker compose up -d redis
	@echo "Starting FastAPI (background)..."
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
	@echo "Starting Streamlit UI..."
	streamlit run src/ui/streamlit_app.py