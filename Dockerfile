# ===== BUILDER STAGE =====
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Copy requirements and install dependencies with build tools
COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y jupyter ipykernel notebook missingno 2>/dev/null || true

# ===== RUNTIME STAGE =====
FROM python:3.11-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    API_URL=http://localhost:8000

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy application code and create directories
COPY app/ ./app/
COPY src/ ./src/
COPY config/ ./config/
COPY frontend/ ./frontend/
COPY data/ ./data/
COPY start.sh .

# Create necessary directories
RUN mkdir -p /app/logs /app/app/model \
    && chmod +x start.sh

# Expose ports (FastAPI=8000, Streamlit=8501)
EXPOSE 8000 8501

# Health check contra a API
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Iniciar ambos os serviços
CMD ["bash", "start.sh"]
