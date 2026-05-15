FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml uv.lock* ./
COPY helpers ./helpers
COPY app ./app

RUN pip install --no-cache-dir uv \
    && uv pip install --system --no-cache -e .

EXPOSE 8000

CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
