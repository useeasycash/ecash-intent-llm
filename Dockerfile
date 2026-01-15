# Stage 1: Builder
FROM python:3.10-slim-bookworm as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
ENV UV_PYTHON_INSTALL_DIR /python
ENV UV_COMPILE_BYTECODE 1
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app
COPY pyproject.toml .

# create a virtual environment and install dependencies
RUN uv venv /opt/venv
# Use the virtual environment's pip
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies (CPU/CUDA agnostic build phase)
# In production, specific torch-cuda wheels might be pinned here for size
RUN uv pip install .

# Stage 2: Runtime
FROM python:3.10-slim-bookworm as runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /bin/uv /bin/uv

# Copy source code
COPY src/ src/
# Copy config
COPY config/ config/

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Entrypoint
CMD ["uvicorn", "ecash_intent_llm.api:app", "--host", "0.0.0.0", "--port", "8000"]
