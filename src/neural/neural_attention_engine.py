# prime-core/neural/neural_attention_engine.py

"""
Neural Attention & Cognitive Control Engine (A142)

--------------------------------------------------

Computes attention-weighted neural context vectors by combining:

- ST window (short-term context)

- MT window (medium-term coherence)

- LT identity (long-term worldview)

This engine produces PRIME's "focus vector" used in reasoning.

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

    def status(self):

        return {

            "focus_dim": self.last_focus_vector.numel() if is_tensor(self.last_focus_vector) else None,

            "focus_preview": self.last_focus_vector[:8].tolist() if is_tensor(self.last_focus_vector) else None

        }

