# ADRAE Docker Compose Orchestration Guide

## Overview

Docker Compose orchestrates ADRAE as a multi-service stack with shared persistent storage and internal networking.

## Services

### adrae-core
- **Purpose**: Runs Prime-Core runtime with MF-500 substrate and S-series pipeline
- **Command**: `runtime`
- **Ports**: None (internal service)
- **Volumes**: Shared persistent storage
- **Restart**: `unless-stopped`

### adrae-health
- **Purpose**: Exposes health monitoring endpoint for external checks
- **Command**: `server`
- **Ports**: `8080:8080` (exposed)
- **Volumes**: Shared persistent storage
- **Restart**: `unless-stopped`
- **Health Check**: Curl-based endpoint probe

## Quick Start

### 1. Build the image
```bash
docker build -t adrae-core:cpu -f docker/Dockerfile .
```

### 2. Start all services
```bash
docker-compose up -d
```

### 3. Check service status
```bash
docker-compose ps
```

### 4. View logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f adrae-core
docker-compose logs -f adrae-health
```

### 5. Stop services
```bash
docker-compose down
```

## Testing

### Health Endpoint
```bash
curl http://localhost:8080/health
```

Expected response:
```json
{
  "status": "ok",
  "zero_norm": 1.0,
  "rand_norm": 1.0,
  "drift": 0.0,
  "timestamp": 1733889999.1234
}
```

## Persistent Storage

All services share the same mounted volumes:

- **Logs**: `/mnt/sage-drive-2/ADRAE/logs` → `/data/logs`
- **Stability**: `/mnt/sage-drive-2/ADRAE/stability` → `/data/stability`
- **Substrate**: `/mnt/sage-drive-2/ADRAE/substrate` → `/data/substrate`

Data persists across:
- Container restarts
- Service updates
- System reboots
- Container recreation

## Network

Services communicate via the `adrae-net` bridge network. This enables:
- Future gateway services
- Internal API communication
- Service discovery
- Kubernetes-ready architecture

## Health Checks

The `adrae-health` service includes automatic health checks:
- **Interval**: 20 seconds
- **Timeout**: 5 seconds
- **Retries**: 3

Docker will automatically restart unhealthy containers.

## Customization

### Change Volume Paths

Edit `docker-compose.yml` and update the volume mounts:
```yaml
volumes:
  - /your/path/ADRAE/logs:/data/logs
  - /your/path/ADRAE/stability:/data/stability
  - /your/path/ADRAE/substrate:/data/substrate
```

### Change Port

To use a different port for the health server:
```yaml
ports:
  - "9000:8080"  # Host:Container
```

### Add Environment Variables

```yaml
environment:
  ADRAE_ENV: production
  ADRAE_LOG_LEVEL: debug
  ADRAE_MODEL_DIM: 128
```

## Next Steps

- **DC-09**: Kubernetes/Talos deployment manifests
- **DC-10**: Image compression and optimization
- **Future**: Gateway services, UI integration, federated nodes

