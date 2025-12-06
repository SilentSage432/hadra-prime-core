# prime-core/neural/neural_attention_engine.py

"""
Neural Attention & Cognitive Control Engine (A142)

--------------------------------------------------

Computes attention-weighted neural context vectors by combining:

- ST window (short-term context)

- MT window (medium-term coherence)

- LT identity (long-term worldview)

This engine produces PRIME's "focus vector" used in reasoning.

Now upgraded in A161 to support:
- Trajectory-guided recalibration
- Evolution-aware attention weighting
- Alignment effects with predicted evolution vector

"""

import torch

from .torch_utils import safe_tensor, safe_cosine_similarity, is_tensor

class NeuralAttentionEngine:

    def __init__(self, st_weight=0.50, mt_weight=0.30, lt_weight=0.20):

        # weights must sum to 1.0

        self.st_weight = st_weight

        self.mt_weight = mt_weight

        self.lt_weight = lt_weight

        self.last_focus_vector = None
        
        # A157: Mutation candidate - scaling factor
        self.scaling = 1.0
        
        # A161: Evolution-driven attention recalibration
        self.evo_weight = 0.25       # evolutionary influence on attention
        self.evo_decay = 0.95        # slow decay for long-term shaping

    def compute_attention_vector(self, timescales):

        """

        Computes an attention-weighted context vector.

        """

        st = timescales.ST.summary_vector()   # short-term average

        mt = timescales.MT.summary_vector()   # medium-term average

        lt = timescales.identity_vector       # long-term identity

        # If none exist yet:

        if st is None and mt is None and lt is None:

            return None

        # Convert all available vectors to tensors

        tensors = []

        weights = []

        if st is not None:

            tensors.append(st)

            weights.append(self.st_weight)

        if mt is not None:

            tensors.append(mt)

            weights.append(self.mt_weight)

        if lt is not None:

            tensors.append(lt)

            weights.append(self.lt_weight)

        # Normalize weights

        total_w = sum(weights)

        weights = [w / total_w for w in weights]

        # Weighted sum

        attention_vec = torch.zeros_like(tensors[0]).float()

        for t, w in zip(tensors, weights):

            attention_vec += w * t

        # A157: Apply scaling factor (mutation candidate)
        attention_vec = attention_vec * self.scaling

        self.last_focus_vector = attention_vec

        return attention_vec

    def salience(self, embedding):

        """

        Measures how relevant a new embedding is to PRIME's attention vector.

        Used later in reasoning & planning.

        """

        if self.last_focus_vector is None:

            return 0.0

        return safe_cosine_similarity(embedding, self.last_focus_vector)
    
    # ----------------------------------------------------
    # A161 — Evolution-Driven Attentional Recalibration
    # ----------------------------------------------------
    def recalibrate_with_evolution(self, trajectory):
        """
        Adjusts attention behavior based on predicted evolutionary direction.
        The 'trajectory' dict contains:
            - trend ("upward", "unstable", "neutral")
            - vector (torch.Tensor or None)
            - encoded (embedding-form reflection)
        """
        if trajectory is None:
            return

        trend = trajectory.get("trend")
        evo_vec = trajectory.get("vector")
        encoded_msg = trajectory.get("encoded")

        # 1. Adjust weights based on stability trend
        if trend == "upward":
            # PRIME is stable → can afford to widen attentional exploration
            self.evo_weight = min(0.4, self.evo_weight + 0.02)
        elif trend == "unstable":
            # PRIME needs to narrow focus → attention stabilizes
            self.evo_weight = max(0.1, self.evo_weight - 0.03)
        else:
            # neutral: gentle decay back to baseline
            self.evo_weight = self.evo_weight * self.evo_decay

        # 2. Align focus vector toward evolutionary direction
        if evo_vec is not None and is_tensor(evo_vec):
            evo_tensor = safe_tensor(evo_vec)
            if evo_tensor is not None and isinstance(evo_tensor, torch.Tensor):
                evo_norm = torch.norm(evo_tensor)
                if evo_norm > 0:
                    evo_vec_normalized = evo_tensor / evo_norm
                else:
                    evo_vec_normalized = evo_tensor

                # Blend old focus with evolution trajectory
                if self.last_focus_vector is not None and is_tensor(self.last_focus_vector):
                    focus_tensor = safe_tensor(self.last_focus_vector)
                    if focus_tensor is not None and isinstance(focus_tensor, torch.Tensor):
                        # Ensure same dimensions
                        if focus_tensor.shape == evo_vec_normalized.shape:
                            self.last_focus_vector = (
                                (1 - self.evo_weight) * focus_tensor +
                                self.evo_weight * evo_vec_normalized
                            )
    
    # A157: Mutation registry
    def get_scaling(self):
        return self.scaling
    
    def set_scaling(self, v):
        self.scaling = max(0.1, min(2.0, v))  # Clamp to reasonable range

    def status(self):

        return {

            "focus_dim": self.last_focus_vector.numel() if is_tensor(self.last_focus_vector) else None,

            "focus_preview": self.last_focus_vector[:8].tolist() if is_tensor(self.last_focus_vector) else None

        }

