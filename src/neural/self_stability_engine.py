# prime-core/neural/self_stability_engine.py

"""
Self-Stability Detection Engine (A155)
--------------------------------------
Monitors PRIME's internal cognitive stability across:

- Drift decay
- Identity vector convergence
- Fusion vector stability
- Reflective coherence
- Semantic anchor consistency

Once all metrics converge for a sustained period,
PRIME can autonomously activate Adaptive Evolution Mode.
"""

import torch
from collections import deque


class SelfStabilityEngine:

    def __init__(self, window=20, drift_threshold=1e-6, similarity_threshold=0.999):
        self.window = window
        self.drift_threshold = drift_threshold
        self.similarity_threshold = similarity_threshold
        
        self.drift_history = deque(maxlen=window)
        self.fusion_history = deque(maxlen=window)
        self.identity_history = deque(maxlen=window)
        self.reflection_history = deque(maxlen=window)
        self.semantic_cluster_history = deque(maxlen=window)

        self.stable_for = 0
        self.required_stable_cycles = 12  # PRIME must remain stable for N cycles

    def _cos(self, a, b):
        if a is None or b is None:
            return 0.0
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            return torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        return 0.0

    def update(self, drift, fusion_vec, identity_vec, reflection_vec, semantic_cluster):
        """
        Drift: float
        fusion_vec: tensor
        identity_vec: tensor
        reflection_vec: tensor
        semantic_cluster: tuple or None
        """
        # 1. Record history
        # Handle None drift by treating as high drift (unstable)
        drift_value = drift if drift is not None else float('inf')
        self.drift_history.append(drift_value)
        self.fusion_history.append(fusion_vec)
        self.identity_history.append(identity_vec)
        self.reflection_history.append(reflection_vec)
        self.semantic_cluster_history.append(semantic_cluster)

        # 2. Check drift stability
        # Filter out None/inf values for stability check
        valid_drifts = [d for d in self.drift_history if d != float('inf') and d is not None]
        drift_stable = (
            len(valid_drifts) == self.window and
            max(valid_drifts) < self.drift_threshold
        )

        # 3. Identity convergence
        if len(self.identity_history) > 2:
            id_sim = self._cos(self.identity_history[-1], self.identity_history[-2])
            identity_stable = id_sim > self.similarity_threshold
        else:
            identity_stable = False

        # 4. Fusion convergence
        if len(self.fusion_history) > 2:
            f_sim = self._cos(self.fusion_history[-1], self.fusion_history[-2])
            fusion_stable = f_sim > self.similarity_threshold
        else:
            fusion_stable = False

        # 5. Reflective coherence
        if len(self.reflection_history) > 2:
            r_sim = self._cos(self.reflection_history[-1], self.reflection_history[-2])
            reflection_stable = r_sim > self.similarity_threshold
        else:
            reflection_stable = False

        # 6. Semantic anchor consistency
        semantic_stable = False
        if len(self.semantic_cluster_history) == self.window:
            # If the top recalled memory stays the same repeatedly
            names = [item[1]["name"] if item and isinstance(item, tuple) and len(item) > 1 and isinstance(item[1], dict) and "name" in item[1] else None for item in self.semantic_cluster_history]
            names = [n for n in names if n is not None]
            if names:
                dominant = max(set(names), key=names.count)
                semantic_stable = names.count(dominant) > self.window * 0.7  # 70% stability

        # 7. Aggregate stability signal
        all_stable = drift_stable and identity_stable and fusion_stable and reflection_stable and semantic_stable

        if all_stable:
            self.stable_for += 1
        else:
            self.stable_for = 0

        ready = self.stable_for >= self.required_stable_cycles

        return {
            "drift_stable": drift_stable,
            "identity_stable": identity_stable,
            "fusion_stable": fusion_stable,
            "reflection_stable": reflection_stable,
            "semantic_stable": semantic_stable,
            "stable_for_cycles": self.stable_for,
            "ready_for_adaptive_evolution": ready
        }

