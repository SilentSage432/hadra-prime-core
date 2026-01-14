# src/observers/phase_observer.py
#
# ⚠️ PHASE I — READ-ONLY OBSERVER ONLY (Python-native)
# This module observes runtime state and suggests a phase.
# It does NOT enforce, block, or influence behavior.
#
# 🔒 PHASE I CONSTRAINT: 
# - No thresholds that change behavior
# - No branching that blocks actions
# - No writing to state
# - No imports into action engine
# - No threads, no new loops
# - Gated by time check in main loop

import time
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from persistence.log_writer import LogWriter


class PhaseObserver:
    """
    Phase Observer — Purely Observational (Python-native)
    
    Observes runtime state and suggests a phase based on current conditions.
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
        
        # ⚠️ PHASE I INVARIANT: This observer never modifies runtime behavior
        # All data structures are for observation purposes only
    
    def observe(self, output: Dict[str, Any], bridge: Any = None) -> Dict[str, Any]:
        """
        Observe runtime state and suggest a phase.
        
        ⚠️ PHASE I CONSTRAINT: This method is purely observational and does not influence behavior.
        
        Args:
            output: The output dict from bridge.cognitive_step() (read-only)
            bridge: Optional bridge reference for reading state (read-only access only)
        
        Returns:
            Dictionary with suggested phase and confidence
        """
        # ⚠️ PHASE I CONSTRAINT: This logic is descriptive only.
        # It does not enforce, block, or gate any actions.
        
        # Default to QUIET_WAKE (the natural state)
        suggested_phase = "QUIET_WAKE"
        confidence = 0.98
        
        # Read-only observation: Check drift metrics
        # This does NOT change behavior, only suggests phase
        drift_data = output.get('drift')
        if drift_data:
            drift_value = None
            if isinstance(drift_data, dict):
                drift_value = drift_data.get('value') or drift_data.get('drift') or drift_data.get('current')
            elif isinstance(drift_data, (int, float)):
                drift_value = drift_data
            
            if drift_value is not None:
                abs_drift = abs(float(drift_value))
                if abs_drift > 0.3:
                    # Observer suggests QUIET_WAKE when drift is significant
                    # This is descriptive only, not enforcement
                    suggested_phase = "QUIET_WAKE"
                    confidence = 0.92
        
        # Read-only observation: Check action presence
        # This does NOT gate actions, only suggests phase
        action = output.get('action')
        if action:
            # Observer may suggest different phase if action present
            # For Phase I, we default to QUIET_WAKE
            suggested_phase = "QUIET_WAKE"
            confidence = 0.90
        
        # Read-only observation: Check fusion state
        # This does NOT modify fusion, only suggests phase
        fusion_data = output.get('fusion')
        if fusion_data:
            # Simple heuristic: if fusion exists, system is active but still QUIET_WAKE
            suggested_phase = "QUIET_WAKE"
            confidence = 0.95
        
        return {
            "phase": suggested_phase,
            "confidence": confidence
        }
    
    def should_emit(self, current_time: float) -> bool:
        """
        Check if 60 seconds have passed since last observation.
        
        ⚠️ PHASE I CONSTRAINT: This is a time gate only, not a behavioral gate.
        
        Args:
            current_time: Current monotonic time
        
        Returns:
            True if 60 seconds have passed, False otherwise
        """
        time_since_last = current_time - self.last_observation_time
        should_emit = time_since_last >= self.observation_interval
        # Diagnostic: Log when we're close to emitting (for debugging)
        if not should_emit and time_since_last >= 55.0:  # Within 5 seconds of emitting
            print(f"[PHASE-I-DEBUG] Phase observer: {time_since_last:.1f}s since last, will emit in {60.0 - time_since_last:.1f}s", flush=True)
        return should_emit
    
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
    def emit(phase_descriptor: Dict[str, Any], observer_output: Dict[str, Any], observer_version: str, log_writer: Optional[LogWriter] = None) -> None:
        """
        Emit phase telemetry metadata to console.
        
        ⚠️ PHASE I CONSTRAINT: This function only writes to telemetry.
        It does not parse, read, or reuse telemetry data.
        
        Args:
            phase_descriptor: Phase descriptor dict (read-only)
            observer_output: Observer output dict (read-only)
            observer_version: Observer version string
        """
        # ⚠️ PHASE I CONSTRAINT: This is write-only telemetry.
        # No runtime logic should read or parse this output.
        
        telemetry_data = {
            "phase": {
                "current": phase_descriptor.get("phase_name", "QUIET_WAKE"),
                "confidence": observer_output.get("confidence", 0.98),
                "authority": phase_descriptor.get("authority", "implicit"),
                "observer_version": observer_version
            }
        }
        
        # Use flush=True to ensure logs appear immediately
        print(f"[ADRAE-PHASE-TELEMETRY] {json.dumps(telemetry_data)}", flush=True)
        # Append-only telemetry to shared log sink if available
        if log_writer:
            try:
                log_writer.write({
                    "source": "phase_i_observer",
                    "telemetry": telemetry_data
                })
            except Exception:
                # Telemetry failures should never disrupt runtime
                pass


def create_default_phase_descriptor() -> Dict[str, Any]:
    """
    Create a default Phase Descriptor for QUIET_WAKE phase.
    
    ⚠️ PHASE I CONSTRAINT: This is a factory function only.
    It does not choose phases or influence behavior.
    
    Returns:
        Default PhaseDescriptor dict with QUIET_WAKE defaults
    """
    now = time.time()
    return {
        "phase_name": "QUIET_WAKE",
        "entered_at": now,
        "last_evaluated_at": now,
        "authority": "implicit",
        "declared_by": "runtime",
        "confidence": 1.0
    }


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
