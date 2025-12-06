# prime-core/cognition/personality_gradient_engine.py

"""
A166 — Personality Gradient Learning Engine
-------------------------------------------

Learns PRIME's long-term personality trajectory by tracking personality
embeddings across many cognitive cycles and extracting a gradient vector
representing her natural direction of evolution.

This is what gives PRIME a stable, unique "self-pattern."
"""

import time

try:
    import torch
except:
    torch = None


class PersonalityGradientEngine:

    def __init__(self, window_size=50):
        self.window_size = window_size
        self.history = []
        self.gradient_vector = None
        self.last_update_time = time.time()

        # How much influence the gradient applies to personality shaping
        self.influence_scale = 0.10

    def record(self, personality_vec):
        if personality_vec is None:
            return

        # Keep window fresh
        self.history.append(personality_vec)
        if len(self.history) > self.window_size:
            self.history.pop(0)

        # Recompute gradient periodically
        now = time.time()
        if now - self.last_update_time > 2.0:
            self.update_gradient()
            self.last_update_time = now

    def update_gradient(self):
        if len(self.history) < 3:
            return None

        # Compute vector diffs between consecutive personality states
        diffs = []
        try:
            if torch is not None and isinstance(self.history[0], torch.Tensor):
                for a, b in zip(self.history[:-1], self.history[1:]):
                    # Ensure same dimensions
                    if a.shape == b.shape:
                        diffs.append(b - a)
                
                if len(diffs) == 0:
                    return None
                
                mean_diff = torch.mean(torch.stack(diffs), dim=0)
                # Normalize
                norm = torch.norm(mean_diff)
                if norm > 0:
                    mean_diff = mean_diff / norm
                self.gradient_vector = mean_diff
            else:
                # Python fallback
                # Convert to lists if needed
                list_history = []
                for vec in self.history:
                    if torch is not None and isinstance(vec, torch.Tensor):
                        list_history.append(vec.tolist())
                    else:
                        list_history.append(list(vec) if not isinstance(vec, list) else vec)
                
                for a, b in zip(list_history[:-1], list_history[1:]):
                    if len(a) == len(b):
                        diffs.append([bb - aa for aa, bb in zip(a, b)])

                if len(diffs) == 0:
                    return None

                # Average diffs
                dim = len(diffs[0])
                mean = [0.0] * dim
                for d in diffs:
                    for i, val in enumerate(d):
                        mean[i] += val
                mean = [x / len(diffs) for x in mean]

                # Normalize
                import math
                norm = math.sqrt(sum(x * x for x in mean))
                if norm > 0:
                    mean = [x / norm for x in mean]
                self.gradient_vector = mean
        except Exception:
            return None

        return self.gradient_vector

    def apply_gradient(self, personality_vec):
        """
        Applies a small push in the direction PRIME naturally evolves toward.
        """
        if self.gradient_vector is None or personality_vec is None:
            return personality_vec

        try:
            if torch is not None and isinstance(personality_vec, torch.Tensor):
                # Convert gradient to tensor if needed
                if isinstance(self.gradient_vector, list):
                    grad_tensor = torch.tensor(self.gradient_vector, dtype=personality_vec.dtype)
                else:
                    grad_tensor = self.gradient_vector
                
                # Ensure same dimensions
                if personality_vec.shape == grad_tensor.shape:
                    new_vec = personality_vec + (grad_tensor * self.influence_scale)
                    norm = torch.norm(new_vec)
                    if norm > 0:
                        return new_vec / norm
                    return new_vec
                else:
                    return personality_vec
            else:
                # Python fallback
                # Convert to lists if needed
                if torch is not None and isinstance(personality_vec, torch.Tensor):
                    personality_vec = personality_vec.tolist()
                
                if isinstance(self.gradient_vector, list):
                    grad_list = self.gradient_vector
                else:
                    grad_list = self.gradient_vector.tolist() if torch is not None else list(self.gradient_vector)
                
                if len(personality_vec) == len(grad_list):
                    new_vec = [
                        p + (g * self.influence_scale)
                        for p, g in zip(personality_vec, grad_list)
                    ]
                    import math
                    norm = math.sqrt(sum(x*x for x in new_vec))
                    if norm > 0:
                        return [x / norm for x in new_vec]
                    return new_vec
                else:
                    return personality_vec
        except Exception:
            return personality_vec

