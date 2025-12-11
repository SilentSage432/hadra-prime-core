# ADRAE Multi-Stage Build Optimization (DC-07)

## Overview

The Dockerfile uses a **multi-stage build** to dramatically reduce image size while maintaining all functionality.

## Build Stages

### Stage 1: Builder
- **Base**: `python:3.12-slim`
- **Purpose**: Install build dependencies and compile Python wheels
- **Tools**: `build-essential`, `git`, `curl`
- **Action**: Installs all Python dependencies (including PyTorch) into `/install`

### Stage 2: Runtime
- **Base**: `python:3.12-slim` (clean slate)
- **Purpose**: Final production container
- **Tools**: Only `curl` (minimal runtime)
- **Action**: Copies compiled dependencies from builder, copies source code

## Size Reduction

| Build Type | Expected Size | Reduction |
|------------|---------------|-----------|
| Before DC-07 (single-stage) | 4-5 GB | - |
| After DC-07 (multi-stage) | 1.2-1.6 GB | **60-75% smaller** |

## What Was Removed

- GCC and build toolchain
- Development headers
- Intermediate build artifacts
- Unnecessary apt packages
- Compiler dependencies

## What Was Preserved

✅ PyTorch 2.2.0 CPU runtime  
✅ All ADRAE Prime-Core dependencies  
✅ MF-500 substrate  
✅ S-series runtime layers  
✅ Health server support  
✅ Persistent volume mounts  
✅ Runtime entrypoint  

## Build Commands

```bash
# Build the optimized image
docker build -t adrae-core:cpu -f docker/Dockerfile .

# Check image size
docker images adrae-core:cpu
```

## Runtime Test

```bash
docker run \
  -p 8080:8080 \
  -v /mnt/sage-drive-2/ADRAE/logs:/data/logs \
  -v /mnt/sage-drive-2/ADRAE/stability:/data/stability \
  -v /mnt/sage-drive-2/ADRAE/substrate:/data/substrate \
  adrae-core:cpu \
  server
```

## Benefits

1. **Faster Deploys**: Smaller images transfer faster
2. **Lower Storage**: Uses less disk space
3. **Better Caching**: Multi-stage builds optimize layer caching
4. **Security**: Smaller attack surface (no build tools in runtime)
5. **Kubernetes Ready**: Optimized for cluster deployment

## Future Enhancements

- DC-10: Image compression (target: ~900 MB)
- CUDA variant: Swap base image for GPU builds
- Alpine variant: Further size reduction (if compatible)

