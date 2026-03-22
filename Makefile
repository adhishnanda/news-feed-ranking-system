install:
	pip install -r requirements.txt

run-redis:
	docker compose up -d redis

ingest:
	python -m src.ingestion.hn_ingest
	python -m src.ingestion.rss_ingest

test:
	pytest