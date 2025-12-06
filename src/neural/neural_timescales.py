# prime-core/neural/neural_timescales.py

"""
Neural Timescales System (A139)

-------------------------------

Maintains PRIME's multi-timescale context windows:

- ST: Short-term (working memory)

- MT: Medium-term (recent patterns)

- LT: Long-term (identity-level averages)

Each timescale stores embeddings and generates summaries.

"""

import torch

from .torch_utils import safe_tensor, is_tensor, safe_cosine_similarity

class TimescaleWindow:

    def __init__(self, name, max_size):

        self.name = name

        self.max_size = max_size

        self.vectors = []

    def add(self, embedding):

        t = safe_tensor(embedding)

        self.vectors.append(t)

        if len(self.vectors) > self.max_size:

            self.vectors.pop(0)

    def summary_vector(self):

        """

        Returns the average embedding or None if empty.

        """

        if len(self.vectors) == 0:

            return None

        try:

            stacked = torch.stack(self.vectors)

            return torch.mean(stacked, dim=0)

        except (RuntimeError, TypeError):

            # Fallback: if stacking fails, return first vector

            return self.vectors[0] if self.vectors else None

    def coherence(self):

        """

        Measures internal similarity within this window.

        """

        if len(self.vectors) < 2:

            return None

        sims = []

        for i in range(len(self.vectors) - 1):

            sims.append(

                safe_cosine_similarity(self.vectors[i], self.vectors[i + 1])

            )

        return sum(sims) / len(sims)

class NeuralTimescales:

    def __init__(self):

        self.ST = TimescaleWindow("short", 5)

        self.MT = TimescaleWindow("medium", 30)

        self.LT = TimescaleWindow("long", 200)

        # Long-term identity anchor vector

        self.identity_vector = None
        
        # A169 — long-horizon identity anchor
        self.long_horizon_identity = None

    def update(self, embedding):

        """

        Push embedding into all timescales and update LT identity.

        """

        if embedding is None:

            return

        t = safe_tensor(embedding)

        # Update windows

        self.ST.add(t)

        self.MT.add(t)

        self.LT.add(t)

        # Update identity vector (EMA)

        if self.identity_vector is None:

            self.identity_vector = t

        else:

            self.identity_vector = 0.98 * self.identity_vector + 0.02 * t
        
        # A169 — Update long-horizon identity (slow-changing)
        if self.long_horizon_identity is None:
            self.long_horizon_identity = t.clone() if hasattr(t, 'clone') else t
        else:
            # slow-moving exponential update (97% old, 3% new)
            try:
                if isinstance(t, torch.Tensor):
                    if isinstance(self.long_horizon_identity, torch.Tensor):
                        if self.long_horizon_identity.shape == t.shape:
                            self.long_horizon_identity = (
                                0.97 * self.long_horizon_identity +
                                0.03 * t
                            )
                    else:
                        # Convert to tensor if needed
                        self.long_horizon_identity = torch.tensor(self.long_horizon_identity, dtype=t.dtype)
                        if self.long_horizon_identity.shape == t.shape:
                            self.long_horizon_identity = (
                                0.97 * self.long_horizon_identity +
                                0.03 * t
                            )
                else:
                    # Python fallback
                    if isinstance(self.long_horizon_identity, torch.Tensor):
                        self.long_horizon_identity = self.long_horizon_identity.tolist()
                    
                    if isinstance(t, torch.Tensor):
                        t = t.tolist()
                    
                    if isinstance(self.long_horizon_identity, list) and isinstance(t, list):
                        if len(self.long_horizon_identity) == len(t):
                            self.long_horizon_identity = [
                                0.97 * old + 0.03 * new
                                for old, new in zip(self.long_horizon_identity, t)
                            ]
            except Exception:
                # If update fails, keep existing long_horizon_identity
                pass

    def summary(self):

        return {

            "ST_dim": self.ST.summary_vector().numel() if self.ST.summary_vector() is not None else None,

            "MT_dim": self.MT.summary_vector().numel() if self.MT.summary_vector() is not None else None,

            "LT_dim": self.LT.summary_vector().numel() if self.LT.summary_vector() is not None else None,

            "ST_coherence": self.ST.coherence(),

            "MT_coherence": self.MT.coherence(),

            "LT_coherence": self.LT.coherence(),

            "identity_vector_preview": self.identity_vector[:8].tolist() if self.identity_vector is not None else None

        }

