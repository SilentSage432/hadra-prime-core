# ============================================================
# ADRAE PRIME-CORE — CPU PyTorch Container
# ============================================================
FROM pytorch/pytorch:2.2.0-cpu

# ------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Create working directory
# ------------------------------------------------------------
WORKDIR /app

# ------------------------------------------------------------
# Copy project files
# ------------------------------------------------------------
COPY . /app/

# ------------------------------------------------------------
# Python dependencies
# ------------------------------------------------------------
COPY docker/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------
ENV PYTHONUNBUFFERED=1
ENV ADRAE_ENV=production
ENV PYTHONPATH=/app/src

# ------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

