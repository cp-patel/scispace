FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    EMBED_CACHE_DIR=/models \
    PATH="/app/.venv/bin:$PATH"

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv
WORKDIR /app

# Dependencies first, so code changes do not reinstall them.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

# Bake the embedding model into the image: containers start fast and never download at runtime.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/models')"

# The screening calibration table, built by evals/selection_eval.py. The worker reads it at start.
COPY evals/results/calibration.json ./evals/results/calibration.json

RUN useradd --create-home app && chown -R app /app /models
USER app

EXPOSE 8000 9100
CMD ["uvicorn", "research_assistant.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
