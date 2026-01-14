# src/observers/phase_observer.py
#
# ⚠️ PHASE I — READ-ONLY OBSERVER ONLY (Python-native)
# This module observes runtime state and infers descriptive phases.
# It emits logs ONLY when the phase changes.
# It does NOT enforce, block, or influence behavior.
#
# 🔒 PHASE I CONSTRAINT: 
# - No thresholds that change behavior
# - No branching that blocks actions
# - No writing to state
# - No imports into action engine
# - No threads, no new loops
# - Gated by time check in main loop
# - Logs only on phase change, not every interval

import time
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from collections import deque
from persistence.log_writer import LogWriter


class PhaseObserver:
    """
    Phase Observer — Purely Observational (Python-native)
    
    Observes runtime state and infers descriptive phases.
    Emits logs ONLY when phase changes (not every interval).
    
    This observer does NOT:
    - Enforce phase transitions
    - Block actions
    - Modify state
    - Influence behavior
    
    ⚠️ PHASE I CONSTRAINT: This is read-only observation only.
    """
    
    def __init__(self):
        # ⚠️ PHASE I: Read-only state tracking
        self.observer_version = "1.0.0-phase-i-python"
        self.last_observation_time = time.monotonic()
        self.observation_interval = 60.0  # 60 seconds
        
        # Track current phase (for change detection)
        # Initialize to None so first observation always triggers a log
        self.current_phase: Optional[str] = None
        
        # Rolling window for metrics (last 60 seconds)
        self.metric_window: deque = deque()  # List of (timestamp, metrics_dict) tuples
        
        # ⚠️ PHASE I INVARIANT: This observer never modifies runtime behavior
        # All data structures are for observation purposes only
    
    def observe(self, output: Dict[str, Any], bridge: Any = None) -> Dict[str, Any]:
        """
        Observe runtime state and infer a phase.
        
        ⚠️ PHASE I CONSTRAINT: This method is purely observational and does not influence behavior.
        
        Args:
            output: The output dict from bridge.cognitive_step() (read-only)
            bridge: Optional bridge reference for reading state (read-only access only)
        
        Returns:
            Dictionary with inferred phase, metrics, and confidence
        """
        # ⚠️ PHASE I CONSTRAINT: This logic is descriptive only.
        # It does not enforce, block, or gate any actions.
        
        current_time = time.monotonic()
        
        # Sample runtime metrics (read-only)
        metrics = self._sample_metrics(output, bridge, current_time)
        
        # Infer phase from metrics
        inferred_phase, confidence = self._infer_phase(metrics)
        
        # Check if phase has changed
        phase_changed = (self.current_phase is None) or (inferred_phase != self.current_phase)
        
        # Store current phase
        self.current_phase = inferred_phase
        
        return {
            "phase": inferred_phase,
            "confidence": confidence,
            "metrics": metrics,
            "phase_changed": phase_changed
        }
    
    def _sample_metrics(self, output: Dict[str, Any], bridge: Any, current_time: float) -> Dict[str, Any]:
        """
        Sample existing runtime metrics (read-only).
        
        ⚠️ PHASE I: This only reads state, never modifies it.
        
        Returns:
            Dictionary with sampled metrics
        """
        metrics = {
            "avg_drift": 0.0,
            "coherence": 1.0,
            "cognitive_step_cadence": 0.0,  # steps per minute
            "idle_action_ratio": 0.0,  # ratio of idle vs active actions
            "timestamp": current_time
        }
        
        # Extract drift value
        drift_data = output.get('drift')
        if drift_data:
            drift_value = None
            if isinstance(drift_data, dict):
                drift_value = drift_data.get('value') or drift_data.get('drift') or drift_data.get('current')
            elif isinstance(drift_data, (int, float)):
                drift_value = drift_data
            
            if drift_value is not None:
                metrics["avg_drift"] = abs(float(drift_value))
        
        # Extract coherence (if available from bridge state)
        if bridge and hasattr(bridge, 'state'):
            try:
                # Try to read coherence from bridge state (read-only)
                if hasattr(bridge.state, 'coherence'):
                    metrics["coherence"] = float(bridge.state.coherence)
                elif hasattr(bridge.state, 'drift') and hasattr(bridge.state.drift, 'get_status'):
                    drift_status = bridge.state.drift.get_status()
                    if isinstance(drift_status, dict):
                        coherence = drift_status.get('coherence', 1.0)
                        metrics["coherence"] = float(coherence) if coherence is not None else 1.0
            except Exception:
                # If coherence unavailable, default to 1.0
                pass
        
        # Calculate cognitive step cadence from metric window
        window_start = current_time - 60.0
        steps_in_window = sum(1 for ts, _ in self.metric_window if ts >= window_start)
        metrics["cognitive_step_cadence"] = steps_in_window  # steps per minute
        
        # Calculate idle vs active action ratio
        actions_in_window = []
        for ts, m in self.metric_window:
            if ts >= window_start:
                action = m.get('action')
                if action:
                    actions_in_window.append(action)
        
        # Classify actions as idle vs active
        idle_actions = ['retrieve_memory', 'analyze_drift']
        active_actions = ['generate_reflection', 'update_identity', 'sync_with_sage', 'propose_thoughts']
        
        idle_count = sum(1 for a in actions_in_window if a in idle_actions)
        active_count = sum(1 for a in actions_in_window if a in active_actions)
        total_classified = idle_count + active_count
        
        if total_classified > 0:
            metrics["idle_action_ratio"] = idle_count / total_classified
        else:
            metrics["idle_action_ratio"] = 1.0  # Default to idle if no actions
        
        # Store action for window
        metrics["action"] = output.get('action')
        
        # Add to metric window
        self.metric_window.append((current_time, metrics))
        
        # Clean old data (keep only last 60 seconds + buffer)
        cutoff_time = current_time - 70.0
        while self.metric_window and self.metric_window[0][0] < cutoff_time:
            self.metric_window.popleft()
        
        return metrics
    
    def _infer_phase(self, metrics: Dict[str, Any]) -> tuple[str, float]:
        """
        Infer descriptive phase from metrics.
        
        ⚠️ PHASE I CONSTRAINT: This is descriptive only, not authoritative.
        The inferred phase does not affect runtime behavior.
        
        Args:
            metrics: Sampled metrics dictionary
        
        Returns:
            Tuple of (phase_name, confidence)
        """
        avg_drift = metrics.get("avg_drift", 0.0)
        coherence = metrics.get("coherence", 1.0)
        cognitive_step_cadence = metrics.get("cognitive_step_cadence", 0)
        idle_action_ratio = metrics.get("idle_action_ratio", 1.0)
        
        # Simple heuristic for Phase I
        # Can be enhanced in future phases without changing behavior
        
        # ACTIVE_CONTINUITY: High cognitive activity with stable metrics
        if cognitive_step_cadence > 50 and idle_action_ratio < 0.5 and coherence > 0.7:
            return ("ACTIVE_CONTINUITY", 0.85)
        
        # QUIET_WAKE: Default natural state (low activity or high idle ratio)
        if cognitive_step_cadence < 30 or idle_action_ratio > 0.7:
            return ("QUIET_WAKE", 0.95)
        
        # QUIET_WAKE: High drift or low coherence suggests quiet state
        if avg_drift > 0.3 or coherence < 0.5:
            return ("QUIET_WAKE", 0.92)
        
        # Default to QUIET_WAKE (the natural state)
        return ("QUIET_WAKE", 0.90)
    
    def should_emit(self, current_time: float) -> bool:
        """
        Check if 60 seconds have passed since last observation.
        
        ⚠️ PHASE I CONSTRAINT: This is a time gate only, not a behavioral gate.
        
        Args:
            current_time: Current monotonic time
        
        Returns:
            True if 60 seconds have passed, False otherwise
        """
        return (current_time - self.last_observation_time) >= self.observation_interval
    
    def update_observation_time(self, current_time: float) -> None:
        """
        Update the last observation time.
        
        ⚠️ PHASE I CONSTRAINT: This only updates internal tracking, no behavior change.
        """
        self.last_observation_time = current_time
    
    def get_version(self) -> str:
        """
        Get observer version for telemetry.
        
        Returns:
            Observer version string
        """
        return self.observer_version


class PhaseTelemetryEmitter:
    """
    Phase Telemetry Emitter — Append-Only (Python-native)
    
    Emits phase telemetry metadata without feedback.
    Telemetry must remain write-only. No parsing or reuse of phase data.
    """
    
    @staticmethod
    def emit_phase_change(
        phase_name: str,
        metrics: Dict[str, Any],
        confidence: float,
        log_writer: Optional[LogWriter] = None
    ) -> None:
        """
        Emit phase change log in the specified format.
        
        ⚠️ PHASE I CONSTRAINT: This function only writes to telemetry.
        It does not parse, read, or reuse telemetry data.
        
        Args:
            phase_name: Name of the phase being entered
            metrics: Sampled metrics dictionary
            confidence: Confidence level (0.0 to 1.0)
            log_writer: Optional LogWriter instance for persistence
        """
        # ⚠️ PHASE I CONSTRAINT: This is write-only telemetry.
        # No runtime logic should read or parse this output.
        
        # Format: [ADRAE-PHASE] ENTER <phase>
        #   avg_drift=<value>
        #   coherence=<value>
        #   Phase_confidence=<value>
        
        avg_drift = metrics.get("avg_drift", 0.0)
        coherence = metrics.get("coherence", 1.0)
        
        # Emit to console (for validation - can be removed later)
        log_line = f"[ADRAE-PHASE] ENTER {phase_name}\n"
        log_line += f"  avg_drift={avg_drift:.2f}\n"
        log_line += f"  coherence={coherence:.2f}\n"
        log_line += f"  Phase_confidence={confidence:.2f}"
        
        print(log_line, flush=True)
        
        # Append-only telemetry to shared log sink
        if log_writer:
            try:
                log_writer.write({
                    "source": "phase_i_observer",
                    "event": "phase_change",
                    "phase": phase_name,
                    "metrics": {
                        "avg_drift": avg_drift,
                        "coherence": coherence,
                        "cognitive_step_cadence": metrics.get("cognitive_step_cadence", 0),
                        "idle_action_ratio": metrics.get("idle_action_ratio", 0.0)
                    },
                    "confidence": confidence
                })
            except Exception:
                # Telemetry failures should never disrupt runtime
                pass


# Singleton instance (read-only observer)
_phase_observer_instance: Optional[PhaseObserver] = None


def get_phase_observer() -> PhaseObserver:
    """
    Get or create the singleton PhaseObserver instance.
    
    ⚠️ PHASE I: One observer instance per process lifetime.
    No threads, no subprocesses.
    """
    global _phase_observer_instance
    if _phase_observer_instance is None:
        _phase_observer_instance = PhaseObserver()
    return _phase_observer_instance


# ⚠️ PHASE I INVARIANT GUARD
# This module must never be imported by:
# - Action engine
# - Arbitration logic
# - Evolution / learning modules
# - Any code that makes behavioral decisions
#
# This observer only suggests phases. It does not enforce them.
# If you see this module used to block or gate actions, STOP and report.
