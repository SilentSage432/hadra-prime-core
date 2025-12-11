# ADRAE Persistent Volume Mounts

## Directory Structure

### Host (SAGE machine 4TB drive):
```
/mnt/sage/ADRAE/
    logs/          → Runtime logs
    stability/     → Drift/norm/stability metrics
    substrate/     → MF-series substrate snapshots (future)
```

### Container (internal paths):
```
/data/logs         → Runtime logs
/data/stability    → Stability snapshots (JSON)
/data/substrate    → Substrate checkpoints
```

## Usage

### Run with volume mounts:
```bash
docker run \
  -p 8080:8080 \
  -v /mnt/sage/ADRAE/logs:/data/logs \
  -v /mnt/sage/ADRAE/stability:/data/stability \
  -v /mnt/sage/ADRAE/substrate:/data/substrate \
  adrae-core:cpu \
  server
```

### Standard runtime (no server mode):
```bash
docker run \
  -v /mnt/sage/ADRAE/logs:/data/logs \
  -v /mnt/sage/ADRAE/stability:/data/stability \
  -v /mnt/sage/ADRAE/substrate:/data/substrate \
  adrae-core:cpu
```

## Persistent Data

- **Stability snapshots**: Written to `/data/stability/stability_<timestamp>.json`
- **Runtime logs**: Written to `/data/logs/`
- **Substrate checkpoints**: Future use in `/data/substrate/`

All data persists across container restarts, redeploys, and crashes.

