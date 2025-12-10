# src/logging/prime_logger.py

import os
import time

LOG_PATH = "prime_runtime.log"
MAX_SIZE_MB = 500        # throttle threshold for dev mode (~500MB)
THROTTLE_INTERVAL = 0.05 # seconds between allowed writes

_last_write = 0

def write_log(message: str):
    global _last_write
    
    # Throttle frequency
    now = time.time()
    if now - _last_write < THROTTLE_INTERVAL:
        return
    
    _last_write = now
    
    # Rotate if size exceeded
    if os.path.exists(LOG_PATH):
        size_mb = os.path.getsize(LOG_PATH) / (1024 * 1024)
        if size_mb > MAX_SIZE_MB:
            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            os.rename(LOG_PATH, f"prime_runtime_{ts}.log")
    
    # Write log entry
    with open(LOG_PATH, "a") as f:
        f.write(message + "\n")

