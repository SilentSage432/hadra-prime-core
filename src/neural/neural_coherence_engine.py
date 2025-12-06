# prime-core/neural/neural_coherence_engine.py

"""
Neural Coherence Engine (A141)

------------------------------

Maintains neural stability, filters noise, and regulates conceptual drift.

Key functions:

- Smooth embeddings

- Anchor to identity vector

- Enforce coherence across timescales

- Reject low-quality neural signals

"""

import torch

from .torch_utils import safe_tensor, safe_cosine_similarity, is_tensor

class NeuralCoherenceEngine:

    def __init__(self, smoothing_factor=0.15, identity_weight=0.25, noise_threshold=0.10):

        self.smoothing_factor = smoothing_factor

        self.identity_weight = identity_weight

        self.noise_threshold = noise_threshold

        self.last_vector = None

    def smooth(self, embedding):

        """

        Exponential smoothing:

        new_vec = (1 - a)*prev + a*current

        """

        t = safe_tensor(embedding)

        if self.last_vector is None:

            self.last_vector = t

            return t

        smoothed = (1 - self.smoothing_factor) * self.last_vector + self.smoothing_factor * t

        self.last_vector = smoothed

        return smoothed

    def anchor_identity(self, vector, identity_vector):

        """

        Blend vector with long-term identity to maintain coherence.

        """

        if identity_vector is None:

            return vector

        iv = safe_tensor(identity_vector)

        v = safe_tensor(vector)

        anchored = (1 - self.identity_weight) * v + self.identity_weight * iv

        return anchored

    def reject_noise(self, prev_vec, new_vec):

        """

        Reject new embeddings if they deviate too sharply from previous ones.

        """

        if prev_vec is None:

            return new_vec

        sim = safe_cosine_similarity(prev_vec, new_vec)

        drift = 1.0 - sim

        if drift > self.noise_threshold:

            # Too much drift → unstable input → reject

            return prev_vec

        return new_vec

    def stabilize(self, embedding, identity_vector=None):

        """

        Full coherence pipeline:

        1. Smooth

        2. Anchor to identity

        3. Reject unstable drift

        A-SOV-06: Supervisory Identity Gate - ADRAE identity vector becomes the stabilizing attractor.

        """

        smoothed = self.smooth(embedding)

        anchored = self.anchor_identity(smoothed, identity_vector)

        stable = self.reject_noise(self.last_vector, anchored)

        # A-SOV-06: Supervisory Identity Gate
        # If embedding diverges too far from ADRAE identity, softly pull it back.
        if identity_vector is not None:
            sim = safe_cosine_similarity(stable, identity_vector)
            if sim is not None and sim < 0.55:
                stable_tensor = safe_tensor(stable)
                identity_tensor = safe_tensor(identity_vector)
                if stable_tensor is not None and identity_tensor is not None:
                    if isinstance(stable_tensor, torch.Tensor) and isinstance(identity_tensor, torch.Tensor):
                        if stable_tensor.shape == identity_tensor.shape:
                            stable = 0.7 * stable_tensor + 0.3 * identity_tensor
                            # Normalize
                            norm = torch.norm(stable)
                            if norm > 0:
                                stable = stable / norm

        # Update last_vector to final stable output for next iteration
        self.last_vector = stable

        return stable

