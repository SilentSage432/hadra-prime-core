# src/prime_core/observable_state.py

"""
READ-ONLY OBSERVABLE STATE EXPORT

This module exposes a minimal, copy-only snapshot of
non-symbolic internal dynamics for external observation.

Hard guarantees:
- No references returned
- No mutation
- No persistence
"""

def export_observable_state(runtime) -> dict:
    """
    Returns a copy-only snapshot of safe scalar observables.

    runtime: PrimeRuntime instance (passed explicitly)
    """
    # Access last output from cognitive_step for scalar observables
    # Use safe access with defaults to prevent errors
    try:
        # Try to access last_output if stored in runtime
        if hasattr(runtime, '_last_output') and runtime._last_output is not None:
            output = runtime._last_output
        else:
            # Fallback to safe defaults if state not yet initialized
            output = None

        if output and isinstance(output, dict):
            # Extract drift (may be a scalar or dict)
            drift_val = output.get('drift', 0.0)
            if isinstance(drift_val, dict):
                drift = float(drift_val.get('magnitude', drift_val.get('value', 0.0)))
            else:
                drift = float(drift_val) if isinstance(drift_val, (int, float)) else 0.0

            # Extract coherence from fusion
            fusion = output.get('fusion', {})
            if isinstance(fusion, dict):
                coherence = float(fusion.get('coherence', 1.0)) if isinstance(fusion.get('coherence'), (int, float)) else 1.0
            else:
                coherence = 1.0

            # Extract entropy from attention
            attention = output.get('attention', {})
            if isinstance(attention, dict):
                entropy = float(attention.get('entropy', 0.0)) if isinstance(attention.get('entropy'), (int, float)) else 0.0
            else:
                entropy = 0.0
        else:
            # Fallback to safe defaults
            drift = 0.0
            coherence = 1.0
            entropy = 0.0
    except Exception:
        # Observer must never affect runtime - use defaults on any error
        drift = 0.0
        coherence = 1.0
        entropy = 0.0

    return {
        "drift": drift,
        "coherence": coherence,
        "attention_entropy": entropy
    }

