# Use a lightweight Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Install production dependencies only
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra-index-url https://download.pytorch.org/whl/cpu

# Download SentenceTransformer model and NLTK data during build
RUN uv run --no-project python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"
RUN uv run --no-project python -m nltk.downloader punkt punkt_tab

# Final stage
FROM python:3.12-slim-bookworm

WORKDIR /app

# Copy installed dependencies and virtual environment
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code and models
COPY app /app/app
COPY helpers /app/helpers
COPY models/current /app/models/current

# Expose port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
