# ============================================================
# STAGE 1 — BUILDER (installs dependencies, compiles wheels)
# ============================================================
FROM python:3.12-slim as builder

# System dependencies for building Python wheels
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy project and requirements
COPY docker/requirements.txt /build/requirements.txt

# Install dependencies into a clean location
RUN pip install --upgrade pip
RUN pip install --prefix=/install -r requirements.txt


# ============================================================
# STAGE 2 — RUNTIME (actual container)
# ============================================================
FROM python:3.12-slim

# Install minimal runtime packages
RUN apt-get update && apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python dependencies from builder
COPY --from=builder /install /usr/local

# Create working directory
WORKDIR /app

# Copy ADRAE source code into runtime
COPY . /app/

# Environment
ENV PYTHONUNBUFFERED=1
ENV ADRAE_ENV=production
ENV PYTHONPATH=/app/src

# Persistent data directories
RUN mkdir -p /data/logs \
             /data/stability \
             /data/substrate

# Entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
