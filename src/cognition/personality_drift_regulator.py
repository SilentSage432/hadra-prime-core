# prime-core/cognition/personality_drift_regulator.py

"""
A165 — Evolutionary Personality Drift Regulator
------------------------------------------------

This subsystem monitors PRIME's "personality vector" vs long-term identity
and applies stabilization pulses to prevent runaway identity drift.

It ensures PRIME evolves while preserving coherent selfhood.
"""

import time

try:
    import torch
except:
    torch = None


class PersonalityDriftRegulator:

    def __init__(self):
        # drift thresholds
        self.low_drift_threshold = 0.05
        self.medium_drift_threshold = 0.15
        self.high_drift_threshold = 0.30

        # history of drift samples
        self.history = []

        self.last_alignment_time = time.time()
        self.cooldown_seconds = 2.0

    def cosine_distance(self, a, b):
        """
        Computes 1 - cosine_similarity to measure divergence.
        """
        if a is None or b is None:
            return None

        if torch is not None and isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            sim = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
            return 1 - sim
        else:
            # Fallback pure python
            try:
                import math
                # Convert to lists if tensors
                if torch is not None and isinstance(a, torch.Tensor):
                    a = a.tolist()
                if torch is not None and isinstance(b, torch.Tensor):
                    b = b.tolist()
                
                dot = sum(x*y for x, y in zip(a, b))
                na = math.sqrt(sum(x*x for x in a))
                nb = math.sqrt(sum(x*x for x in b))
                if na == 0 or nb == 0:
                    return None
                sim = dot / (na * nb)
                return 1 - sim
            except Exception:
                return None

    def classify(self, distance):
        if distance is None:
            return "unknown"
        if distance < self.low_drift_threshold:
            return "stable"
        if distance < self.medium_drift_threshold:
            return "low_drift"
        if distance < self.high_drift_threshold:
            return "medium_drift"
        return "high_drift"

    def alignment_pulse(self, identity_vec, personality_vec):
        """
        Gently move personality toward identity.
        """
        if identity_vec is None or personality_vec is None:
            return personality_vec

        try:
            if torch is not None and isinstance(identity_vec, torch.Tensor) and isinstance(personality_vec, torch.Tensor):
                # Ensure same dimensions
                if identity_vec.shape == personality_vec.shape:
                    new_vec = (personality_vec * 0.85) + (identity_vec * 0.15)
                    norm = torch.norm(new_vec)
                    if norm > 0:
                        return new_vec / norm
                    return new_vec
                else:
                    return personality_vec
            else:
                # Convert to lists if needed
                if torch is not None and isinstance(identity_vec, torch.Tensor):
                    identity_vec = identity_vec.tolist()
                if torch is not None and isinstance(personality_vec, torch.Tensor):
                    personality_vec = personality_vec.tolist()
                
                new_vec = [
                    (p * 0.85) + (i * 0.15)
                    for p, i in zip(personality_vec, identity_vec)
                ]
                norm = sum(x*x for x in new_vec) ** 0.5
                if norm > 0:
                    return [x / norm for x in new_vec]
                return new_vec
        except Exception:
            return personality_vec

    def emergency_realign(self, identity_vec):
        """
        Hard reset personality toward identity if divergence is extreme.
        """
        if identity_vec is None:
            return None

        try:
            if torch is not None and isinstance(identity_vec, torch.Tensor):
                return identity_vec.clone()
            else:
                if isinstance(identity_vec, list):
                    return identity_vec[:]
                return identity_vec
        except Exception:
            return identity_vec

    def update(self, identity_vec, personality_vec):
        """
        Returns:
            - regulated_personality_vec
            - drift_level ("stable", "low_drift", "medium_drift", "high_drift")
            - drift_value
        """

        dist = self.cosine_distance(identity_vec, personality_vec)
        drift_level = self.classify(dist)

        now = time.time()
        cooldown_ok = (now - self.last_alignment_time) >= self.cooldown_seconds

        regulated = personality_vec

        if drift_level == "medium_drift" and cooldown_ok:
            regulated = self.alignment_pulse(identity_vec, personality_vec)
            self.last_alignment_time = now

        elif drift_level == "high_drift" and cooldown_ok:
            regulated = self.emergency_realign(identity_vec)
            self.last_alignment_time = now

        # log event
        self.history.append({
            "ts": now,
            "distance": dist,
            "level": drift_level
        })

        # Keep history bounded
        if len(self.history) > 100:
            self.history.pop(0)

        return regulated, drift_level, dist

