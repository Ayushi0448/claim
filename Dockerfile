# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Build tools are needed for scipy/scikit-learn wheels on some architectures.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so the layer caches across source edits.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY scripts/ ./scripts/
COPY evaluation/ ./evaluation/
COPY frontend/ ./frontend/
COPY data/policy/ ./data/policy/

# Build the index at image build time so the container starts ready. The index
# is a pure function of the policy PDF, so baking it in is safe and removes
# several seconds from cold start on a free-tier host.
RUN python -m scripts.ingest_policy

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys,os; \
url=f'http://127.0.0.1:{os.getenv(\"PORT\",\"8000\")}/health'; \
sys.exit(0 if urllib.request.urlopen(url, timeout=8).status == 200 else 1)"

# Render and most PaaS providers inject $PORT at runtime.
CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
