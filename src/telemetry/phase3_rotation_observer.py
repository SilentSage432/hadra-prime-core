# src/telemetry/phase3_rotation_observer.py
#
# ⚠️ PHASE III — ROTATION BASELINES & WINDOW DETECTION (READ-ONLY)
# This module detects rotation windows (stable QUIET_WAKE spans) and builds
# baseline statistics over hours/days — without changing cognition or cadence.
#
# 🔒 PHASE III CONSTRAINTS:
# - Read-only observation
# - Append-only logs (JSONL)
# - No feedback into cognition
# - No changes to action weights, loop cadence, or behavior
# - Low-frequency writes (10 min for baselines, on window open/close)
# - Must survive restarts safely

import time
import json
import os
from datetime import datetime, timezone
from collections import defaultdict, deque
from typing import Dict, Any, Optional

# Persistent data root directory
DATA_ROOT = "/data"


class Phase3RotationObserver:
    """
    Phase III — Rotation Baselines & Window Detection Observer
    
    Detects rotation windows (stable QUIET_WAKE spans) and builds baseline
    statistics over time. Purely observational, no behavior changes.
    """
    
    def __init__(
        self,
        data_dir: str = "/data/observations",
        baseline_emit_interval_sec: float = 600.0,  # 10 minutes
        window_min_duration_sec: float = 1800.0,  # 30 minutes (minimum for window to count)
        quiet_step_threshold: int = 3,  # <=3 steps/min means quiet
        stability_required: bool = True
    ):
        # ⚠️ PHASE III: Read-only state tracking
        self.data_dir = data_dir
        self.baseline_emit_interval_sec = baseline_emit_interval_sec
        self.window_min_duration_sec = window_min_duration_sec
        self.quiet_step_threshold = quiet_step_threshold
        self.stability_required = stability_required
        
        # Ensure observations directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # File paths for JSONL persistence
        self.windows_file = os.path.join(self.data_dir, "phase3_rotation_windows.jsonl")
        self.baselines_file = os.path.join(self.data_dir, "phase3_rotation_baselines.jsonl")
        
        # Rolling buffer for rhythm samples (last 60 samples = ~60 minutes if sampled ~1/min)
        # Each entry: (timestamp, rhythm_payload_dict)
        self.rhythm_buffer: deque = deque(maxlen=60)
        
        # Current "open window" state (None if no window open)
        self.current_window: Optional[Dict[str, Any]] = None
        
        # EWMA baselines for steps/min and action proportions
        self.ewma_steps_per_min: float = 0.0
        self.ewma_alpha: float = 0.1  # EWMA smoothing factor
        self.action_histogram_ewma: Dict[str, float] = defaultdict(float)
        
        # Window statistics (for baseline tracking)
        self.total_windows_detected: int = 0
        self.window_durations: deque = deque(maxlen=100)  # Keep last 100 window durations
        
        # Restart settle time tracking
        self.process_start_time: float = time.time()
        self.last_restart_settle_estimate: Optional[float] = None
        
        # Last emit timestamps
        self.last_baseline_emit_time: float = time.monotonic()
        self.last_rhythm_time: float = 0.0
        
        # Stability detection (requires 3 consecutive samples)
        self.stability_samples: deque = deque(maxlen=3)
        
        # One-time initial log flag (for deployment verification)
        self._initial_log_emitted: bool = False
        
        # ⚠️ PHASE III INVARIANT: This observer never modifies runtime behavior
        # All data structures are for observation purposes only
    
    def observe_step(self, step_output: Dict[str, Any]) -> None:
        """
        Observe a single cognitive step.
        
        ⚠️ PHASE III CONSTRAINT: This method only records data.
        It does not modify output, delay execution, or gate actions.
        
        Args:
            step_output: The output dict from bridge.cognitive_step()
        """
        # This method is primarily for extracting per-step data if needed
        # Phase III primarily consumes Phase II rhythm payloads via observe_rhythm()
        # But we can extract action names for window tracking
        
        # Extract action if present (for window state tracking)
        action = step_output.get('action')
        if action and self.current_window is not None:
            # Update window action histogram
            self.current_window["action_histogram"][action] = \
                self.current_window["action_histogram"].get(action, 0) + 1
    
    def observe_rhythm(self, rhythm_payload: Dict[str, Any]) -> None:
        """
        Observe a Phase II rhythm payload (preferred input).
        
        ⚠️ PHASE III CONSTRAINT: This method only processes rhythm data.
        It does not modify runtime behavior.
        
        Args:
            rhythm_payload: Phase II rhythm telemetry payload dict
        """
        current_time = time.monotonic()
        wall_time = time.time()
        
        # Extract rhythm metrics
        inferred_state = rhythm_payload.get("inferred_state", "QUIET_WAKE")
        cognitive_steps_last_minute = rhythm_payload.get("cognitive_steps_last_minute", 0)
        avg_drift = rhythm_payload.get("avg_drift", 0.0)
        coherence = rhythm_payload.get("coherence", 1.0)
        actions_last_minute = rhythm_payload.get("actions_last_minute", {})
        uptime_seconds = rhythm_payload.get("uptime_seconds", 0)
        
        # ⚠️ PHASE III: One-time initial log (deployment verification only)
        if not self._initial_log_emitted:
            print("[ADRAE-ROTATION] Phase III observer active", flush=True)
            self._initial_log_emitted = True
        
        # Add to rhythm buffer
        self.rhythm_buffer.append((current_time, rhythm_payload))
        
        # Update EWMA baselines
        self._update_baselines(cognitive_steps_last_minute, actions_last_minute)
        
        # Update stability samples for window detection
        is_stable = self._check_stability(
            inferred_state,
            cognitive_steps_last_minute,
            coherence,
            avg_drift
        )
        self.stability_samples.append(is_stable)
        
        # Check window open/close conditions
        self._update_window_state(
            current_time,
            wall_time,
            uptime_seconds,
            inferred_state,
            cognitive_steps_last_minute,
            coherence,
            avg_drift
        )
        
        # Check if baseline should be emitted (every 10 minutes)
        self._maybe_emit_baseline(current_time, wall_time, uptime_seconds)
        
        self.last_rhythm_time = current_time
    
    def tick(self, now_ts: float) -> None:
        """
        Periodic tick handler (safe to call frequently).
        
        ⚠️ PHASE III CONSTRAINT: This is a time check only, not a behavioral gate.
        
        Args:
            now_ts: Current monotonic time
        """
        # This method is called every step but internally rate-limits operations
        # Window detection and baseline emits are handled in observe_rhythm()
        # This method is reserved for any future time-based operations
        pass
    
    def _check_stability(
        self,
        inferred_state: str,
        cognitive_steps: int,
        coherence: float,
        avg_drift: float
    ) -> bool:
        """
        Check if current rhythm sample indicates stability for window opening.
        
        Window OPEN condition requires all to be true:
        - inferred_state == "QUIET_WAKE"
        - cognitive_steps_last_minute <= quiet_step_threshold
        - coherence >= 0.99
        - avg_drift <= 0.15
        
        Returns:
            True if sample meets stability criteria
        """
        if inferred_state != "QUIET_WAKE":
            return False
        
        if cognitive_steps > self.quiet_step_threshold:
            return False
        
        if coherence < 0.99:
            return False
        
        if avg_drift > 0.15:
            return False
        
        return True
    
    def _update_window_state(
        self,
        current_time: float,
        wall_time: float,
        uptime_seconds: int,
        inferred_state: str,
        cognitive_steps: int,
        coherence: float,
        avg_drift: float
    ) -> None:
        """
        Update window open/close state based on stability conditions.
        
        Window OPEN: requires 3 consecutive stable samples (3 minutes)
        Window CLOSE: any instability condition holds for 2 consecutive samples
        """
        # Check if we have 3 consecutive stable samples
        has_three_stable = (
            len(self.stability_samples) >= 3 and
            all(self.stability_samples)
        )
        
        # Check if we have 2 consecutive unstable samples
        has_two_unstable = (
            len(self.stability_samples) >= 2 and
            not any(self.stability_samples[-2:])
        )
        
        # Window OPEN condition
        if self.current_window is None and has_three_stable:
            self._open_window(current_time, wall_time, uptime_seconds)
        
        # Window CLOSE conditions
        elif self.current_window is not None:
            should_close = False
            
            # Close if inferred_state != "QUIET_WAKE" for 2 samples
            if inferred_state != "QUIET_WAKE" and has_two_unstable:
                should_close = True
            
            # Close if cognitive_steps > 5 for 2 samples
            if cognitive_steps > 5 and has_two_unstable:
                should_close = True
            
            # Close if coherence drops
            if coherence < 0.99 and has_two_unstable:
                should_close = True
            
            # Close if drift exceeds threshold
            if avg_drift > 0.15 and has_two_unstable:
                should_close = True
            
            if should_close:
                self._close_window(current_time, wall_time)
        
        # Update current window metrics if window is open
        if self.current_window is not None:
            self.current_window["cognitive_steps"].append(cognitive_steps)
            self.current_window["last_update_time"] = current_time
    
    def _open_window(self, current_time: float, wall_time: float, uptime_seconds: int) -> None:
        """Open a new rotation window."""
        self.current_window = {
            "start_timestamp": datetime.fromtimestamp(wall_time, tz=timezone.utc).isoformat(),
            "start_monotonic": current_time,
            "start_uptime_seconds": uptime_seconds,
            "cognitive_steps": [],
            "action_histogram": defaultdict(int),
            "last_update_time": current_time
        }
        
        # Emit window open log
        log_data = {
            "timestamp": self.current_window["start_timestamp"],
            "uptime_seconds": uptime_seconds,
            "steps_per_min": 0,  # Will be populated on close
            "confidence": 0.95
        }
        
        print(f"[ADRAE-ROTATION-WINDOW] OPEN {json.dumps(log_data)}", flush=True)
    
    def _close_window(self, current_time: float, wall_time: float) -> None:
        """Close current rotation window and persist to JSONL."""
        if self.current_window is None:
            return
        
        # Calculate window statistics
        duration_sec = current_time - self.current_window["start_monotonic"]
        cognitive_steps = self.current_window["cognitive_steps"]
        mean_steps = sum(cognitive_steps) / len(cognitive_steps) if cognitive_steps else 0.0
        peak_steps = max(cognitive_steps) if cognitive_steps else 0
        
        # Only record window if it meets minimum duration
        if duration_sec >= self.window_min_duration_sec:
            self.total_windows_detected += 1
            self.window_durations.append(duration_sec)
            
            # Prepare window record
            window_record = {
                "start_timestamp": self.current_window["start_timestamp"],
                "end_timestamp": datetime.fromtimestamp(wall_time, tz=timezone.utc).isoformat(),
                "duration_sec": round(duration_sec, 2),
                "start_uptime_seconds": self.current_window["start_uptime_seconds"],
                "mean_steps_per_min": round(mean_steps, 2),
                "peak_steps_per_min": peak_steps,
                "action_histogram": dict(self.current_window["action_histogram"]),
                "restart_events": 0  # Will be enhanced in future phases
            }
            
            # Write to JSONL file (append-only)
            try:
                with open(self.windows_file, "a") as f:
                    f.write(json.dumps(window_record) + "\n")
            except Exception as e:
                print(f"[PHASE-III-ERROR] Failed to write window record: {e}", flush=True)
            
            # Emit window close log
            log_data = {
                "duration_sec": round(duration_sec, 2),
                "mean_steps": round(mean_steps, 2),
                "action_mix": dict(self.current_window["action_histogram"])
            }
            
            print(f"[ADRAE-ROTATION-WINDOW] CLOSE {json.dumps(log_data)}", flush=True)
        
        # Reset current window
        self.current_window = None
    
    def _update_baselines(
        self,
        cognitive_steps: int,
        actions_last_minute: Dict[str, int]
    ) -> None:
        """Update EWMA baselines for steps/min and action proportions."""
        # Update EWMA steps per minute
        if self.ewma_steps_per_min == 0.0:
            self.ewma_steps_per_min = float(cognitive_steps)
        else:
            self.ewma_steps_per_min = (
                self.ewma_alpha * cognitive_steps +
                (1.0 - self.ewma_alpha) * self.ewma_steps_per_min
            )
        
        # Update EWMA action proportions
        total_actions = sum(actions_last_minute.values())
        if total_actions > 0:
            for action, count in actions_last_minute.items():
                proportion = count / total_actions
                if self.action_histogram_ewma[action] == 0.0:
                    self.action_histogram_ewma[action] = proportion
                else:
                    self.action_histogram_ewma[action] = (
                        self.ewma_alpha * proportion +
                        (1.0 - self.ewma_alpha) * self.action_histogram_ewma[action]
                    )
    
    def _maybe_emit_baseline(self, current_time: float, wall_time: float, uptime_seconds: int) -> None:
        """Emit baseline snapshot every 10 minutes if interval has passed."""
        time_since_last_baseline = current_time - self.last_baseline_emit_time
        
        if time_since_last_baseline >= self.baseline_emit_interval_sec:
            self._emit_baseline(current_time, wall_time, uptime_seconds)
            self.last_baseline_emit_time = current_time
    
    def _emit_baseline(self, current_time: float, wall_time: float, uptime_seconds: int) -> None:
        """Emit baseline snapshot and persist to JSONL."""
        # Calculate average window duration
        avg_window_duration = (
            sum(self.window_durations) / len(self.window_durations)
            if self.window_durations else None
        )
        
        # Get top 5 actions by EWMA proportion
        top_actions = sorted(
            self.action_histogram_ewma.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Prepare baseline record
        baseline_record = {
            "timestamp": datetime.fromtimestamp(wall_time, tz=timezone.utc).isoformat(),
            "uptime_seconds": uptime_seconds,
            "ewma_steps_per_min": round(self.ewma_steps_per_min, 2),
            "top_5_action_proportions": {action: round(prop, 4) for action, prop in top_actions},
            "total_windows_detected": self.total_windows_detected,
            "average_window_duration_sec": round(avg_window_duration, 2) if avg_window_duration else None,
            "last_restart_settle_estimate_sec": self.last_restart_settle_estimate
        }
        
        # Write to JSONL file (append-only)
        try:
            with open(self.baselines_file, "a") as f:
                f.write(json.dumps(baseline_record) + "\n")
        except Exception as e:
            print(f"[PHASE-III-ERROR] Failed to write baseline record: {e}", flush=True)
        
        # Emit baseline log
        print(f"[ADRAE-ROTATION-BASELINE] {json.dumps(baseline_record)}", flush=True)


# Singleton instance (read-only observer)
_phase3_observer_instance: Optional[Phase3RotationObserver] = None


def get_phase3_rotation_observer() -> Phase3RotationObserver:
    """
    Get or create the singleton Phase3RotationObserver instance.
    
    ⚠️ PHASE III: One observer instance per process lifetime.
    No threads, no subprocesses.
    """
    global _phase3_observer_instance
    if _phase3_observer_instance is None:
        _phase3_observer_instance = Phase3RotationObserver()
    return _phase3_observer_instance


# ⚠️ PHASE III INVARIANT GUARD
# This module must never be imported by:
# - Cognition logic
# - Action selection engines
# - Arbitration logic
# - Evolution / learning modules
# - Any code that makes behavioral decisions
#
# Rotation observation is read-only and append-only.
# If you see this module used to gate, delay, or modify behavior, STOP and report.
