# src/cognition/internal_events.py

"""
Internal Events Logger

Silent internal events for ADRAE's cognitive processes.
Not surfaced to users, only logged for introspection.
"""

import os
import json
import time
from typing import Optional

DATA_ROOT = "/data/events"
EVENTS_LOG = os.path.join(DATA_ROOT, "internal.log")

# Event types
EVENT_OBSERVATION_REPEAT = "observation_repeat"
EVENT_PATTERN_DENSITY_INCREASE = "pattern_density_increase"
EVENT_INFERENCE_WEAKENED = "inference_weakened"
EVENT_INFERENCE_STABILIZED = "inference_stabilized"

class InternalEventsLogger:
    """
    Logs silent internal cognitive events.
    No user-facing output, only internal diagnostics.
    """
    
    def __init__(self):
        os.makedirs(DATA_ROOT, exist_ok=True)
        # Ensure log file exists
        if not os.path.exists(EVENTS_LOG):
            with open(EVENTS_LOG, "w") as f:
                pass
    
    def log_event(
        self,
        event_type: str,
        details: Optional[dict] = None
    ):
        """
        Log an internal event silently.
        Never affects runtime behavior.
        """
        try:
            entry = {
                "timestamp": time.time(),
                "event": event_type,
                "details": details or {}
            }
            
            with open(EVENTS_LOG, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            # Silent failure - events must not affect runtime
            pass

# Global instance
_internal_events_logger = None

def get_event_logger() -> InternalEventsLogger:
    """Get or create global event logger instance."""
    global _internal_events_logger
    if _internal_events_logger is None:
        _internal_events_logger = InternalEventsLogger()
    return _internal_events_logger

