# prime-core/cognition/supervisory_control_network.py

# ============================================
# A179 — Supervisory Control Network (SCN)
# ============================================
# Oversees PRIME's entire cognition system:
# - evaluates thoughts
# - aligns cognition with identity & intentions
# - suppresses unhelpful patterns
# - adjusts action probabilities
# - stabilizes drift
# - enforces cognitive purpose

try:
    import torch
except ImportError:
    torch = None

from ..neural.torch_utils import safe_tensor, safe_cosine_similarity, TORCH_AVAILABLE


class SupervisoryControlNetwork:
    """
    A179 — Supervisory Control Network (SCN)

    Oversees PRIME's entire cognition system:

    - evaluates thoughts
    - aligns cognition with identity & intentions
    - suppresses unhelpful patterns
    - adjusts action probabilities
    - stabilizes drift
    - enforces cognitive purpose
    """

    def __init__(self):
        # multipliers that dynamically regulate cognition
        self.action_bias = {
            "retrieve_memory": 1.0,
            "generate_reflection": 1.0,
            "analyze_drift": 1.0,
            "update_identity": 1.0,
            "sync_with_sage": 1.0,
            "propose_thoughts": 1.0,
            "reinforce_attention": 1.0,
            "evolve": 1.0,
            "enter_adaptive_evolution": 1.0,
        }

    def evaluate_thought(self, vec, state):
        """
        Score a thought for:
        - identity alignment
        - coherence with workspace
        - drift risk

        Returns:
            float: Score between 0.0 and 1.0 (higher = better)
        """
        if vec is None:
            return 0.5

        try:
            vec_tensor = safe_tensor(vec)
            if vec_tensor is None:
                return 0.5

            # Get identity vector
            identity = None
            if hasattr(state, 'timescales') and state.timescales is not None:
                identity = getattr(state.timescales, 'identity_vector', None)

            if identity is None:
                return 0.5

            identity_tensor = safe_tensor(identity)
            if identity_tensor is None:
                return 0.5

            # Compute alignment
            alignment = safe_cosine_similarity(vec_tensor, identity_tensor)
            if alignment is None:
                alignment = 0.5
            else:
                alignment = float(alignment)

            # Get drift penalty
            drift_penalty = 0.0
            if hasattr(state, 'drift') and state.drift is not None:
                try:
                    if hasattr(state.drift, 'avg_drift'):
                        avg_drift = state.drift.avg_drift
                        if avg_drift is not None:
                            drift_penalty = float(avg_drift) * 2.0
                    elif hasattr(state.drift, 'get_status'):
                        drift_status = state.drift.get_status()
                        if drift_status and isinstance(drift_status, dict):
                            latest_drift = drift_status.get("latest_drift", 0.0)
                            drift_penalty = float(latest_drift) * 2.0
                except Exception:
                    pass

            # Score = alignment - drift_penalty, clamped to [0, 1]
            score = max(0.0, min(1.0, alignment - drift_penalty))
            return score

        except Exception:
            # If evaluation fails, return neutral score
            return 0.5

    def adjust_action_bias(self, thought_score):
        """
        Reward or penalize cognitive actions based on thought quality.

        High-quality thoughts (score > 0.7):
        - Reinforce stable, aligned thought patterns
        - Boost memory and reflection

        Low-quality thoughts (score <= 0.7):
        - Penalize unstable or misaligned thoughts
        - Reduce evolution and random thought generation
        """
        if thought_score > 0.7:
            # reinforce stable, aligned thought patterns
            self.action_bias["generate_reflection"] *= 1.05
            self.action_bias["retrieve_memory"] *= 1.05
            self.action_bias["update_identity"] *= 1.02
        else:
            # penalize unstable or misaligned thoughts
            self.action_bias["evolve"] *= 0.9
            self.action_bias["propose_thoughts"] *= 0.95
            self.action_bias["enter_adaptive_evolution"] *= 0.9

        # clamp values to prevent extreme biases
        for k in self.action_bias:
            self.action_bias[k] = float(min(max(self.action_bias[k], 0.1), 3.0))

    def supervise(self, chosen_embedding, state):
        """
        Apply supervisory logic each cognitive cycle.

        Args:
            chosen_embedding: The selected thought vector
            state: Cognitive state object

        Returns:
            dict with:
                - "score": float (thought quality score)
                - "action_bias": dict (adjusted action biases)
        """
        score = self.evaluate_thought(chosen_embedding, state)
        self.adjust_action_bias(score)

        return {
            "score": score,
            "action_bias": dict(self.action_bias),
        }

