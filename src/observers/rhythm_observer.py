# src/observers/rhythm_observer.py
#
# ⚠️ PHASE II — RHYTHM OBSERVER (READ-ONLY)
# This module passively records ADRAE's natural cognitive + runtime rhythm.
#
# 🔒 PHASE II CONSTRAINTS:
# - Read-only observation
# - Append-only logs
# - No feedback into cognition
# - No gating, no delays, no sleeps injected
# - Lives entirely in Python runtime
# - No threads, no subprocesses, no timers that fight the main loop

import time
import json
from datetime import datetime, timezone
from collections import defaultdict, deque
from typing import Dict, Any, Optional
from persistence.log_writer import LogWriter


class RhythmObserver:
    """
    Phase II — Rhythm Observer
    
    Passively records ADRAE's natural cognitive + runtime rhythm once per minute.
    Uses monotonic time to track 60-second intervals without interfering with the main loop.
    """
    
    def __init__(self, phase3_observer=None):
        # ⚠️ PHASE II: Read-only state tracking
        self.start_time = time.monotonic()  # Monotonic time for interval tracking
        self.process_start_time = time.time()  # Wall-clock time for uptime calculation
        self.last_observation_time = self.start_time
        
        # Rolling window data (last 60 seconds)
        self.action_counts: deque = deque()  # List of (timestamp, action_name) tuples
        self.drift_values: deque = deque()  # List of (timestamp, drift_value) tuples
        self.cognitive_step_timestamps: deque = deque()  # Timestamps of cognitive steps
        
        # Current window boundaries
        self.observation_interval = 60.0  # 60 seconds
        
        # ⚠️ PHASE III: Optional hook for Phase III observer (preferred)
        self.phase3_observer = phase3_observer
        
        # ⚠️ PHASE II INVARIANT: This observer never modifies runtime behavior
        # All data structures are append-only for observation purposes
    
    def observe_step(self, output: Dict[str, Any], log_writer: Optional[LogWriter] = None) -> None:
        """
        Observe a single cognitive step.
        
        ⚠️ PHASE II CONSTRAINT: This method only records data.
        It does not modify output, delay execution, or gate actions.
        
        Args:
            output: The output dict from bridge.cognitive_step()
        """
        current_time = time.monotonic()
        wall_time = time.time()
        
        # Record cognitive step timestamp
        self.cognitive_step_timestamps.append((current_time, wall_time))
        
        # Record action if present
        action = output.get('action')
        if action:
            self.action_counts.append((current_time, action))
        
        # Record drift if present
        drift_data = output.get('drift')
        if drift_data:
            # Extract drift value (handle various formats)
            drift_value = None
            if isinstance(drift_data, dict):
                drift_value = drift_data.get('value') or drift_data.get('drift') or drift_data.get('current')
            elif isinstance(drift_data, (int, float)):
                drift_value = drift_data
            
            if drift_value is not None:
                self.drift_values.append((current_time, float(drift_value)))
        
        # Clean old data (keep only last 60 seconds + buffer)
        cutoff_time = current_time - 70.0  # 70 second buffer for safety
        self._clean_old_data(cutoff_time)
        
        # Check if 60 seconds have passed since last observation
        time_since_last = current_time - self.last_observation_time
        if time_since_last >= self.observation_interval:
            self._emit_rhythm_log(current_time, wall_time, log_writer)
            self.last_observation_time = current_time
        # Diagnostic: Log when we're close to emitting (for debugging)
        elif time_since_last >= 55.0:  # Within 5 seconds of emitting
            print(f"[PHASE-II-DEBUG] Rhythm observer: {time_since_last:.1f}s since last, will emit in {60.0 - time_since_last:.1f}s", flush=True)
    
    def _clean_old_data(self, cutoff_time: float) -> None:
        """
        Remove data older than cutoff_time from rolling windows.
        
        ⚠️ PHASE II: This is housekeeping only, not behavioral logic.
        """
        # Clean action counts
        while self.action_counts and self.action_counts[0][0] < cutoff_time:
            self.action_counts.popleft()
        
        # Clean drift values
        while self.drift_values and self.drift_values[0][0] < cutoff_time:
            self.drift_values.popleft()
        
        # Clean cognitive step timestamps
        while self.cognitive_step_timestamps and self.cognitive_step_timestamps[0][0] < cutoff_time:
            self.cognitive_step_timestamps.popleft()
    
    def _emit_rhythm_log(self, current_monotonic: float, current_wall: float, log_writer: Optional[LogWriter] = None) -> None:
        """
        Emit a single [ADRAE-RHYTHM] log line.
        
        ⚠️ PHASE II CONSTRAINT: This is write-only telemetry.
        No runtime logic should read or parse this output.
        
        Args:
            current_monotonic: Current monotonic time
            current_wall: Current wall-clock time
        """
        # Calculate uptime
        uptime_seconds = int(current_monotonic - self.start_time)
        since_last_restart_seconds = uptime_seconds  # Same for now (no restart tracking yet)
        
        # Count cognitive steps in last 60 seconds
        window_start = current_monotonic - 60.0
        cognitive_steps_last_minute = sum(
            1 for ts, _ in self.cognitive_step_timestamps
            if ts >= window_start
        )
        
        # Count actions in last 60 seconds
        action_counts_dict = defaultdict(int)
        for ts, action in self.action_counts:
            if ts >= window_start:
                action_counts_dict[action] += 1
        
        # Calculate drift statistics
        drift_window = [drift for ts, drift in self.drift_values if ts >= window_start]
        avg_drift = sum(drift_window) / len(drift_window) if drift_window else 0.0
        latest_drift = drift_window[-1] if drift_window else 0.0
        
        # Infer coherence (simplified - can be enhanced later)
        # For Phase II, use a simple heuristic based on drift stability
        if drift_window:
            drift_variance = sum((d - avg_drift) ** 2 for d in drift_window) / len(drift_window)
            coherence = max(0.0, min(1.0, 1.0 - (drift_variance * 10)))  # Simple coherence estimate
        else:
            coherence = 1.0
        
        # Infer state (descriptive only, not authoritative)
        inferred_state = self._infer_state(cognitive_steps_last_minute, avg_drift, coherence)
        
        # Build log payload
        rhythm_data = {
            "timestamp": datetime.fromtimestamp(current_wall, tz=timezone.utc).isoformat(),
            "uptime_seconds": uptime_seconds,
            "since_last_restart_seconds": since_last_restart_seconds,
            "cognitive_steps_last_minute": cognitive_steps_last_minute,
            "actions_last_minute": dict(action_counts_dict),
            "avg_drift": round(avg_drift, 6),
            "latest_drift": round(latest_drift, 6),
            "coherence": round(coherence, 3),
            "inferred_state": inferred_state
        }
        
        # ⚠️ PHASE II: Write-only telemetry emission
        # Use flush=True to ensure logs appear immediately
        print(f"[ADRAE-RHYTHM] {json.dumps(rhythm_data)}", flush=True)
        # Append-only telemetry to shared log sink if available
        if log_writer:
            try:
                log_writer.write({
                    "source": "phase_ii_rhythm",
                    "telemetry": rhythm_data
                })
            except Exception:
                # Telemetry failures should never disrupt runtime
                pass
        
        # ⚠️ PHASE III: Optional hook to feed rhythm payload to Phase III observer (preferred)
        if self.phase3_observer:
            try:
                self.phase3_observer.observe_rhythm(rhythm_data)
            except Exception:
                # Phase III failures should never disrupt Phase II or runtime
                pass
    
    def _infer_state(self, cognitive_steps: int, avg_drift: float, coherence: float) -> str:
        """
        Infer descriptive state from observed metrics.
        
        ⚠️ PHASE II CONSTRAINT: This is descriptive only, not authoritative.
        The inferred state does not affect runtime behavior.
        
        Args:
            cognitive_steps: Number of cognitive steps in last minute
            avg_drift: Average drift value
            coherence: Coherence estimate
        
        Returns:
            Descriptive state string (e.g., "QUIET_WAKE")
        """
        # Simple heuristic for Phase II
        # Can be enhanced in future phases without changing behavior
        
        if cognitive_steps == 0:
            return "QUIET_WAKE"
        
        if avg_drift > 0.3 or coherence < 0.5:
            return "DRIFT_ELEVATED"
        
        if cognitive_steps > 100:  # High activity
            return "ACTIVE"
        
        # Default to QUIET_WAKE (the natural state)
        return "QUIET_WAKE"


# Singleton instance (one per process lifetime)
_rhythm_observer_instance: Optional[RhythmObserver] = None


def get_rhythm_observer(phase3_observer=None) -> RhythmObserver:
    """
    Get or create the singleton RhythmObserver instance.
    
    ⚠️ PHASE II: One observer instance per process lifetime.
    No threads, no subprocesses.
    
    Args:
        phase3_observer: Optional Phase III observer instance to receive rhythm payloads
    """
    global _rhythm_observer_instance
    if _rhythm_observer_instance is None:
        _rhythm_observer_instance = RhythmObserver(phase3_observer=phase3_observer)
    else:
        # Update phase3_observer if provided after first initialization
        if phase3_observer is not None:
            _rhythm_observer_instance.phase3_observer = phase3_observer
    return _rhythm_observer_instance


# ⚠️ PHASE II INVARIANT GUARD
# This module must never be imported by:
# - Cognition logic
# - Action selection engines
# - Arbitration logic
# - Evolution / learning modules
# - Any code that makes behavioral decisions
#
# Rhythm observation is read-only and append-only.
# If you see this module used to gate, delay, or modify behavior, STOP and report.
