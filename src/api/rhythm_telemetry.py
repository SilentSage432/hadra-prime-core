# src/api/rhythm_telemetry.py
#
# ⚠️ PHASE A — RHYTHM TELEMETRY ENDPOINT (READ-ONLY)
# This module exposes a minimal HTTP endpoint for observing ADRAE's rhythm state.
#
# 🔒 PHASE A CONSTRAINTS:
# - Read-only observation
# - No side effects
# - No logging
# - No authentication
# - No background threads
# - No memory changes

from datetime import datetime, timezone
from typing import Optional, Dict, Any
from fastapi import FastAPI
from fastapi.responses import JSONResponse


def create_rhythm_endpoint(runtime_getter):
    """
    Create a FastAPI app with the /api/adrae/rhythm endpoint.
    
    ⚠️ PHASE A: This endpoint is read-only and has no side effects.
    
    Args:
        runtime_getter: Function that returns the PrimeRuntime instance (or None)
    
    Returns:
        FastAPI app instance
    """
    app = FastAPI()
    
    @app.get("/api/adrae/rhythm")
    def get_rhythm():
        """
        GET /api/adrae/rhythm
        
        Returns ADRAE's self-reported rhythm state.
        
        ⚠️ PHASE A: This endpoint is read-only and has no side effects.
        """
        try:
            runtime = runtime_getter()
            
            # If runtime is unavailable, return unavailable state
            if runtime is None:
                return JSONResponse(content={
                    "state": "unavailable",
                    "cadence_ms": 30000,
                    "last_tick": None,
                    "continuity": {
                        "window": "rolling",
                        "depth": "bounded",
                        "retention": "non-accumulative"
                    },
                    "notes": None
                })
            
            # Get rhythm observer
            rhythm_observer = None
            if hasattr(runtime, 'rhythm_observer'):
                rhythm_observer = runtime.rhythm_observer
            
            # Determine state from runtime and rhythm observer
            state = _determine_state(runtime, rhythm_observer)
            
            # Get cadence (30 seconds = 30000ms)
            cadence_ms = 30000
            
            # Get last tick timestamp
            last_tick = _get_last_tick(runtime, rhythm_observer)
            
            # Build response (exact schema)
            response = {
                "state": state,
                "cadence_ms": cadence_ms,
                "last_tick": last_tick,
                "continuity": {
                    "window": "rolling",
                    "depth": "bounded",
                    "retention": "non-accumulative"
                },
                "notes": None
            }
            
            return JSONResponse(content=response)
            
        except Exception:
            # On any error, return unavailable state
            return JSONResponse(content={
                "state": "unavailable",
                "cadence_ms": 30000,
                "last_tick": None,
                "continuity": {
                    "window": "rolling",
                    "depth": "bounded",
                    "retention": "non-accumulative"
                },
                "notes": None
            })
    
    return app


def _determine_state(runtime: Any, rhythm_observer: Optional[Any]) -> str:
    """
    Determine ADRAE's self-reported state.
    
    ⚠️ PHASE A: This reads state only, does not modify anything.
    
    State values: "idle" | "active" | "focused" | "degraded" | "unavailable"
    """
    # If runtime is not running, return unavailable
    if not hasattr(runtime, 'running') or not runtime.running:
        return "unavailable"
    
    # If runtime has no bridge, return unavailable
    if not hasattr(runtime, 'bridge') or runtime.bridge is None:
        return "unavailable"
    
    # Check if we have recent cognitive steps
    if rhythm_observer is not None:
        # Check if we have cognitive step timestamps
        if hasattr(rhythm_observer, 'cognitive_step_timestamps'):
            import time
            current_time = time.monotonic()
            window_start = current_time - 60.0  # Last 60 seconds
            
            # Count steps in last minute
            recent_steps = sum(
                1 for ts, _ in rhythm_observer.cognitive_step_timestamps
                if ts >= window_start
            )
            
            # If no recent steps, state is idle
            if recent_steps == 0:
                return "idle"
            
            # Check drift to determine if degraded
            if hasattr(rhythm_observer, 'drift_values'):
                drift_window = [
                    drift for ts, drift in rhythm_observer.drift_values
                    if ts >= window_start
                ]
                if drift_window:
                    avg_drift = sum(drift_window) / len(drift_window)
                    if avg_drift > 0.3:
                        return "degraded"
            
            # If we have steps and low drift, determine active vs focused
            # For Phase A, we'll use a simple heuristic:
            # - If steps > 50 per minute: active
            # - If steps > 0 and <= 50: focused
            if recent_steps > 50:
                return "active"
            else:
                return "focused"
    
    # Default to idle if we can't determine from rhythm observer
    return "idle"


def _get_last_tick(runtime: Any, rhythm_observer: Optional[Any]) -> Optional[str]:
    """
    Get the last tick timestamp as ISO-8601 string.
    
    ⚠️ PHASE A: This reads state only, does not modify anything.
    """
    if rhythm_observer is not None:
        # Get the most recent cognitive step timestamp
        if hasattr(rhythm_observer, 'cognitive_step_timestamps'):
            if rhythm_observer.cognitive_step_timestamps:
                # Get the last (most recent) timestamp
                _, wall_time = rhythm_observer.cognitive_step_timestamps[-1]
                # Convert to ISO-8601
                return datetime.fromtimestamp(wall_time, tz=timezone.utc).isoformat()
        
        # Fallback: use last observation time
        if hasattr(rhythm_observer, 'last_observation_time'):
            import time
            # Convert monotonic to wall time (approximate)
            # This is a fallback, so we'll use current time as approximation
            current_wall = time.time()
            return datetime.fromtimestamp(current_wall, tz=timezone.utc).isoformat()
    
    # If we can't determine, return None
    return None
