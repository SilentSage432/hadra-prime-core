# prime-core/cognition/supervisory_conflict_resolver.py

# ============================================
# A180 — Supervisory Conflict Resolution Layer (SCRL)
# ============================================
# Detects internal conflicts between:
# - evolution impulses vs. stability requirements
# - identity alignment vs. novelty
# - task focus vs. reflection
# - short-term vs. long-term coherence
#
# Generates a "conflict vector" and final resolution score.

try:
    import torch
except ImportError:
    torch = None

from ..neural.torch_utils import safe_tensor, safe_cosine_similarity, TORCH_AVAILABLE


class SupervisoryConflictResolver:
    """
    A180 — Supervisory Conflict Resolution Layer (SCRL)

    Detects internal conflicts between:

    - evolution impulses vs. stability requirements
    - identity alignment vs. novelty
    - task focus vs. reflection
    - short-term vs. long-term coherence

    Generates a "conflict vector" and final resolution score.
    """

    def __init__(self):
        self.last_conflict = None

    def detect_conflict(self, state, bridge=None):
        """
        Analyze drift, identity alignment, evolution pressure, and workspace tension.

        Args:
            state: Cognitive state object
            bridge: Optional bridge object for accessing fusion vector

        Returns:
            dict: Conflict metrics including drift, identity alignment, evolution pressure
        """
        # Get drift metrics
        drift = 0.0
        avg_drift = 0.0
        try:
            if hasattr(state, 'drift') and state.drift is not None:
                if hasattr(state.drift, 'get_status'):
                    drift_status = state.drift.get_status()
                    if drift_status and isinstance(drift_status, dict):
                        drift = float(drift_status.get("latest_drift", 0.0))
                
                if hasattr(state.drift, 'avg_drift'):
                    avg_drift_val = state.drift.avg_drift
                    if avg_drift_val is not None:
                        avg_drift = float(avg_drift_val)
        except Exception:
            pass

        # Get identity and fusion vectors
        identity = None
        fusion = None
        
        try:
            if hasattr(state, 'timescales') and state.timescales is not None:
                identity = getattr(state.timescales, 'identity_vector', None)
        except Exception:
            pass

        try:
            if bridge is not None and hasattr(bridge, 'fusion'):
                fusion = getattr(bridge.fusion, 'last_fusion_vector', None)
        except Exception:
            pass

        # Compute identity alignment
        id_align = 0.5  # Default neutral alignment
        if identity is not None and fusion is not None:
            try:
                identity_tensor = safe_tensor(identity)
                fusion_tensor = safe_tensor(fusion)
                if identity_tensor is not None and fusion_tensor is not None:
                    alignment = safe_cosine_similarity(identity_tensor, fusion_tensor)
                    if alignment is not None:
                        id_align = float(alignment)
            except Exception:
                pass

        # Get evolution pressure (may be stored in state or default to neutral)
        evolve_bias = 0.5
        try:
            if hasattr(state, 'evolution_pressure'):
                evolve_bias = float(state.evolution_pressure)
        except Exception:
            pass

        conflict = {
            "drift": drift,
            "avg_drift": avg_drift,
            "identity_alignment": id_align,
            "evolution_pressure": evolve_bias,
        }

        return conflict

    def resolve(self, conflict):
        """
        Produce a resolution score:

        - high score (0.7+) → safe to evolve or take creative actions
        - medium score (0.4-0.7) → balanced approach
        - low score (<0.4) → enforce stability actions

        Args:
            conflict: dict from detect_conflict()

        Returns:
            float: Resolution score between 0.0 and 1.0
        """
        id_align = conflict.get("identity_alignment", 0.5)
        drift = conflict.get("drift", 0.0)
        avg_drift = conflict.get("avg_drift", 0.0)
        evolve = conflict.get("evolution_pressure", 0.5)

        # Base resolution: weighted combination of alignment and stability
        score = id_align * 0.6 + (1.0 - min(drift, 1.0)) * 0.4

        # If evolution pressure is extremely high but alignment is low → suppress
        if evolve > 0.7 and id_align < 0.4:
            score *= 0.7

        # If drift is rising → stability is favored
        if drift > 0.1:
            score *= (1.0 - min(drift, 0.5))  # Cap drift impact at 0.5

        # If average drift is high, further favor stability
        if avg_drift > 0.15:
            score *= 0.9

        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))

        self.last_conflict = conflict

        return score

