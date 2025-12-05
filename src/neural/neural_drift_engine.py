# prime-core/neural/neural_drift_engine.py

"""
Neural Drift Detection Engine (A138)

------------------------------------

Tracks changes in PRIME's neural embeddings over time

to detect conceptual drift, stability, and coherence.

Drift is measured using cosine distance between

sequential embeddings and aggregated moving averages.

"""

import time

import torch

from .torch_utils import safe_cosine_similarity, safe_tensor

class NeuralDriftEngine:

    def __init__(self, window_size=20):

        self.window_size = window_size

        self.history = []             # list of (timestamp, tensor)

        self.drift_scores = []        # list of floats

        self.avg_drift = 0.0

    def record(self, embedding):

        """

        Add a new embedding snapshot to the drift engine.

        """

        if embedding is None:

            return

        tensor = safe_tensor(embedding)

        ts = time.time()

        self.history.append((ts, tensor))

        # limit size

        if len(self.history) > self.window_size:

            self.history.pop(0)

        # compute drift after adding

        self._compute_drift()

    def _compute_drift(self):

        """

        Calculates drift between latest and previous embedding.

        Drift = 1 - cosine_similarity

        """

        if len(self.history) < 2:

            return

        _, prev = self.history[-2]

        _, curr = self.history[-1]

        sim = safe_cosine_similarity(prev, curr)

        drift = 1.0 - sim  # higher drift = greater conceptual change

        self.drift_scores.append(drift)

        # compute moving average

        if len(self.drift_scores) > 50:

            self.drift_scores.pop(0)

        self.avg_drift = sum(self.drift_scores) / len(self.drift_scores)

    def get_status(self):

        """

        Returns drift-related diagnostic information.

        """

        return {

            "history_length": len(self.history),

            "latest_drift": self.drift_scores[-1] if self.drift_scores else None,

            "avg_drift": self.avg_drift,

            "window": self.window_size,

        }

