FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY scripts ./scripts
COPY data ./data
COPY models ./models

# Install uv and dependencies
RUN pip install uv && uv sync --frozen

# Train model if not exists
RUN if [ ! -f models/lgbm_credit_default.joblib ]; then \
    uv run python scripts/train.py; \
    fi

# Expose metrics and API ports
EXPOSE 8000

# Run FastAPI
CMD ["uv", "run", "uvicorn", "self_healing_pipeline.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
