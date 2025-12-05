# prime-core/neural/neural_state_tracker.py

"""
Neural State Tracker

--------------------

Central monitoring system for neural activity.

Aggregates:

- latest embedding

- drift engine metrics

- embedding stats

- system-level coherence signals (added later)

"""

from .neural_drift_engine import NeuralDriftEngine
from .neural_timescales import NeuralTimescales

from .torch_utils import is_tensor

class NeuralStateTracker:

    def __init__(self):

        self.last_embedding = None
        self.last_perception = None

        self.drift = NeuralDriftEngine()
        self.timescales = NeuralTimescales()

    def update(self, embedding):

        """

        Update global neural state and push into drift engine.

        """

        if embedding is None:

            return

        self.last_embedding = embedding

        self.drift.record(embedding)
        self.timescales.update(embedding)

    def summary(self):

        emb = self.last_embedding

        return {

            "last_embedding_dim": emb.numel() if emb is not None and is_tensor(emb) else None,

            "drift": self.drift.get_status(),

            "timescales": self.timescales.summary()

        }

