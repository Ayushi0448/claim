.PHONY: help install ingest api ui test eval lint clean docker-up docker-down

help:
	@echo "install     Install dependencies"
	@echo "ingest      Build the policy index"
	@echo "api         Run the FastAPI backend on :8000"
	@echo "ui          Run the Streamlit frontend on :8501"
	@echo "test        Run the test suite"
	@echo "eval        Run the full evaluation"
	@echo "docker-up   Build and start API + frontend via compose"

install:
	pip install -r requirements.txt

ingest:
	python -m scripts.ingest_policy --inspect

api:
	uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

ui:
	streamlit run frontend/streamlit_app.py

test:
	python -m pytest tests/ -v

eval:
	python -m evaluation.run_evaluation

lint:
	ruff check app evaluation scripts tests

clean:
	rm -rf data/index __pycache__ .pytest_cache
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

docker-up:
	docker compose up --build

docker-down:
	docker compose down
