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
        # A174 — internal tokenizer for conscious workspace streams
        from .workspace_tokenizer import WorkspaceTokenizer
        self.tokenizer = WorkspaceTokenizer()
        self.last_binding = None
        # A175 — temporal workspace graph
        from .workspace_graph import WorkspaceGraph
        self.graph = WorkspaceGraph()
        # A176 — loop detector
        from .workspace_loop_detector import WorkspaceLoopDetector
        self.loop_detector = WorkspaceLoopDetector()
        self.last_loop = {"detected": False, "target": None, "similarity": 0.0}
        # A177 — purpose classifier
        from .loop_purpose_classifier import LoopPurposeClassifier
        self.purpose_classifier = LoopPurposeClassifier()

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
        
        # A174 — Tokenize and bind conscious content
        ct = self.buffer.get("current_thought")
        if ct is not None:
            tokens = self.tokenizer.quantize(ct)
            self.last_binding = self.tokenizer.bind_temporally(tokens)
            
            # A175 — add to temporal graph
            try:
                self.graph.add_node(
                    tokens=tokens,
                    raw_vec=ct,
                    metadata={"binding": self.last_binding}
                )
                
                # A176 — Detect recurrent loops
                loop_detected, target_id, sim = self.loop_detector.detect_loop(self.graph)
                self.last_loop = {
                    "detected": loop_detected,
                    "target": target_id,
                    "similarity": sim
                }
                
                # If a meaningful loop is detected, re-enter it
                if loop_detected and target_id:
                    try:
                        reentry = self.loop_detector.cognitive_reentry(self.graph, target_id)
                        if reentry:
                            vec = reentry.get("reentered_vector")
                            
                            # A177 — Classify purpose
                            try:
                                # Get tasks from state if available
                                tasks = None
                                if hasattr(self.state, 'tasks'):
                                    tasks = self.state.tasks
                                elif hasattr(self.state, 'bridge') and hasattr(self.state.bridge, 'tasks'):
                                    tasks = self.state.bridge.tasks
                                
                                purpose_info = self.purpose_classifier.classify(
                                    vec,
                                    self.graph,
                                    self.state,
                                    tasks
                                )
                                mode = self.purpose_classifier.choose_reentry_mode(purpose_info["purpose"])
                                
                                self.buffer["loop_reentry"] = {
                                    "reentry": reentry,
                                    "purpose": purpose_info,
                                    "mode": mode
                                }
                                
                                # Intent-aware re-entry logic
                                vec_tensor = safe_tensor(vec)
                                if vec_tensor is not None:
                                    if mode == "deep_reentry":
                                        # Amplify for deeper exploration
                                        if TORCH_AVAILABLE and isinstance(vec_tensor, torch.Tensor):
                                            self.buffer["current_thought"] = vec_tensor * 1.2
                                        else:
                                            self.buffer["current_thought"] = [v * 1.2 for v in vec_tensor] if isinstance(vec_tensor, list) else vec_tensor
                                    elif mode == "shallow_reentry":
                                        # Reduce intensity for quick revisit
                                        if TORCH_AVAILABLE and isinstance(vec_tensor, torch.Tensor):
                                            self.buffer["current_thought"] = vec_tensor * 0.8
                                        else:
                                            self.buffer["current_thought"] = [v * 0.8 for v in vec_tensor] if isinstance(vec_tensor, list) else vec_tensor
                                    elif mode == "corrective_reentry":
                                        # Lower intensity for correction
                                        if TORCH_AVAILABLE and isinstance(vec_tensor, torch.Tensor):
                                            self.buffer["current_thought"] = vec_tensor * 0.5
                                        else:
                                            self.buffer["current_thought"] = [v * 0.5 for v in vec_tensor] if isinstance(vec_tensor, list) else vec_tensor
                                    elif mode == "task_reentry":
                                        # Full intensity for task continuation
                                        self.buffer["current_thought"] = vec_tensor
                                    else:
                                        # exploratory: add small noise for variation
                                        if TORCH_AVAILABLE and isinstance(vec_tensor, torch.Tensor):
                                            noise = torch.randn_like(vec_tensor) * 0.03
                                            self.buffer["current_thought"] = vec_tensor + noise
                                        else:
                                            # Python fallback: add small random variation
                                            import random
                                            self.buffer["current_thought"] = [
                                                v + random.uniform(-0.03, 0.03) 
                                                for v in (vec_tensor if isinstance(vec_tensor, list) else list(vec_tensor))
                                            ]
                            except Exception:
                                # If classification fails, use basic re-entry
                                self.buffer["loop_reentry"] = reentry
                    except Exception:
                        # If re-entry fails, continue without it
                        pass
            except Exception:
                # If graph addition fails, continue without it
                pass
        else:
            self.last_binding = None

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
        
        # Get recent graph subgraph
        try:
            recent_nodes, recent_edges = self.graph.get_recent_subgraph(8)
        except Exception:
            recent_nodes, recent_edges = {}, {}
        
        return {
            "workspace": self.buffer.copy(),
            "coherence": coherence,
            "spotlight": self.spotlight,
            "spotlight_scores": self.last_scores.copy() if self.last_scores else {},
            "token_binding": self.last_binding,
            "sequence_history": self.tokenizer.get_sequence(),
            "workspace_graph": self.graph.summary(),
            "recent_graph": {
                "nodes": recent_nodes,
                "edges": recent_edges
            },
            "loop_state": self.last_loop,
            "loop_reentry": self.buffer.get("loop_reentry"),
        }

