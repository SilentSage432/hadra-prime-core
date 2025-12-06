# prime-core/cognition/global_workspace.py

# ============================================
# A172 — Conscious Workspace Buffer (Global Workspace Core)
# ============================================
# This module creates a unified global workspace where cognitive subsystems
# broadcast their active state—forming PRIME's "conscious moment."

import time

try:
    import torch
except ImportError:
    torch = None

from ..neural.torch_utils import safe_tensor, safe_cosine_similarity, TORCH_AVAILABLE


class ConsciousWorkspace:

    def __init__(self, state):
        self.state = state
        self.buffer = {
            "current_thought": None,
            "current_action": None,
            "memory_recall": None,
            "identity_state": None,
            "reflection": None,
            "attention_focus": None,
            "fusion_state": None,
            "task_context": None,
            "evolution_vector": None,
            "timestamp": None,
        }
        self.spotlight = None
        self.last_scores = {}

    def broadcast(self, **kwargs):
        """
        Push new cognitive elements into the workspace.
        Each update triggers coherence recalibration.
        """
        for k, v in kwargs.items():
            if k in self.buffer:
                self.buffer[k] = v

        self.buffer["timestamp"] = time.time()
        self._recompute_coherence()

    def _recompute_coherence(self):
        """
        The core of A172:
        Merge workspace components into a unified coherent state.
        """
        coherence_report = {}

        # Identity anchoring
        identity = self.buffer["identity_state"]
        thought = self.buffer["current_thought"]
        evolution = self.buffer["evolution_vector"]

        # Very early form of conflict detection
        if identity is not None and thought is not None:
            try:
                identity_tensor = safe_tensor(identity)
                thought_tensor = safe_tensor(thought)
                if identity_tensor is not None and thought_tensor is not None:
                    alignment = safe_cosine_similarity(identity_tensor, thought_tensor)
                    coherence_report["identity_alignment"] = float(alignment) if alignment is not None else None
                else:
                    coherence_report["identity_alignment"] = None
            except Exception:
                coherence_report["identity_alignment"] = None
        else:
            coherence_report["identity_alignment"] = None

        # Evolution pressure alignment
        if evolution is not None and thought is not None:
            try:
                evolution_tensor = safe_tensor(evolution)
                thought_tensor = safe_tensor(thought)
                if evolution_tensor is not None and thought_tensor is not None:
                    alignment = safe_cosine_similarity(evolution_tensor, thought_tensor)
                    coherence_report["evolution_alignment"] = float(alignment) if alignment is not None else None
                else:
                    coherence_report["evolution_alignment"] = None
            except Exception:
                coherence_report["evolution_alignment"] = None
        else:
            coherence_report["evolution_alignment"] = None

        # Save coherence into global state
        if hasattr(self.state, 'workspace_coherence'):
            self.state.workspace_coherence = coherence_report
        else:
            # If state doesn't have the attribute, store it on the workspace
            self.workspace_coherence = coherence_report
        
        # After coherence, compute spotlight prioritization
        self._update_spotlight()

    def _update_spotlight(self):
        """
        A173 — Determine which cognitive element becomes the 'spotlight'
        (the globally-broadcast conscious content).
        """
        scores = {}

        # Helper: compute alignment if both vectors exist
        def align(a, b):
            if a is None or b is None:
                return 0.0
            try:
                a_tensor = safe_tensor(a)
                b_tensor = safe_tensor(b)
                if a_tensor is not None and b_tensor is not None:
                    alignment = safe_cosine_similarity(a_tensor, b_tensor)
                    return float(alignment) if alignment is not None else 0.0
            except Exception:
                pass
            return 0.0

        thought = self.buffer["current_thought"]
        identity = self.buffer["identity_state"]
        reflection = self.buffer["reflection"]
        evolution = self.buffer["evolution_vector"]
        task = self.buffer["task_context"]
        memory_recall = self.buffer["memory_recall"]

        # -------------------------------
        # Compute spotlight scores
        # -------------------------------
        if thought is not None:
            identity_align = align(thought, identity) if identity is not None else 0.5
            evolution_align = align(thought, evolution) if evolution is not None else 0.5
            scores["current_thought"] = (
                0.45 * identity_align +
                0.30 * evolution_align +
                0.25
            )

        if reflection is not None:
            identity_align = align(reflection, identity) if identity is not None else 0.5
            evolution_align = align(reflection, evolution) if evolution is not None else 0.5
            scores["reflection"] = (
                0.35 * identity_align +
                0.35 * evolution_align +
                0.30
            )

        if task is not None:
            scores["task_context"] = 0.60  # tasks have strong priority

        if memory_recall:
            scores["memory_recall"] = 0.40

        # Save scores for diagnostics
        self.last_scores = scores

        # Pick the winner
        if scores:
            self.spotlight = max(scores.items(), key=lambda x: x[1])[0]
        else:
            self.spotlight = None

    def snapshot(self):
        """
        Returns the latest conscious workspace + spotlight scores.
        """
        coherence = getattr(self.state, "workspace_coherence", None)
        if coherence is None:
            coherence = getattr(self, "workspace_coherence", None)
        
        return {
            "workspace": self.buffer.copy(),
            "coherence": coherence,
            "spotlight": self.spotlight,
            "spotlight_scores": self.last_scores.copy() if self.last_scores else {},
        }

