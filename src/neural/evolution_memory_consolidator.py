# prime-core/neural/evolution_memory_consolidator.py

"""
Evolutionary Memory Consolidator (A159)
--------------------------------------
Transforms individual evolution events into durable long-term
narrative memories that reinforce PRIME's developmental identity.

This module:
- Extracts recent evolution events
- Builds semantic summaries
- Updates long-term identity vectors
- Stores consolidated memories into neural semantic memory
"""

try:
    import torch
except ImportError:
    torch = None

from .torch_utils import safe_tensor, TORCH_AVAILABLE


class EvolutionMemoryConsolidator:

    def __init__(self, window=10):
        self.window = window  # number of evolution events to summarize
        self.buffer = []      # temporary holding area

    def record(self, evo_event):
        """Store raw evolution event."""
        if evo_event:
            self.buffer.append(evo_event)
            # Keep window size small
            if len(self.buffer) > self.window:
                self.buffer.pop(0)

    def consolidate(self, memory_manager, hooks, timescales):
        """Transform raw events into narrative long-term memory."""

        if not self.buffer:
            return None

        # Build text summary
        summary_lines = []
        for event in self.buffer:
            if event.get("evolved"):
                old_val = event.get('old', 0)
                new_val = event.get('new', 0)
                if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                    summary_lines.append(
                        f"I improved coherence by evolving from {old_val:.4f} to {new_val:.4f}."
                    )
                else:
                    summary_lines.append(
                        "I improved coherence through internal evolution."
                    )
            elif event.get("rolled_back"):
                old_val = event.get('old', 0)
                new_attempt = event.get('new_attempt', 0)
                if isinstance(old_val, (int, float)) and isinstance(new_attempt, (int, float)):
                    summary_lines.append(
                        f"I attempted evolution from {old_val:.4f} "
                        f"to {new_attempt:.4f} but reverted to maintain stability."
                    )
                else:
                    summary_lines.append(
                        "I attempted evolution but reverted to maintain stability."
                    )

        summary_text = " ".join(summary_lines)

        if not summary_text.strip():
            return None

        # Encode into embedding
        embedding = hooks.on_reflection(summary_text)

        # Store as semantic narrative concept
        if memory_manager and embedding is not None:
            import time
            concept_name = f"evo_summary_{int(time.time())}"
            memory_manager.store_concept(concept_name, embedding)

        # Update identity vector slightly toward this summary
        if timescales and hasattr(timescales, "identity_vector"):
            try:
                iv = timescales.identity_vector
                embedding_tensor = safe_tensor(embedding)
                iv_tensor = safe_tensor(iv)
                
                if TORCH_AVAILABLE and isinstance(iv_tensor, torch.Tensor) and isinstance(embedding_tensor, torch.Tensor):
                    # Blend: 90% current identity, 10% new evolutionary insight
                    timescales.identity_vector = (iv_tensor * 0.9 + embedding_tensor * 0.1)
                    # Normalize to maintain vector magnitude
                    norm = torch.norm(timescales.identity_vector)
                    if norm > 0:
                        timescales.identity_vector = timescales.identity_vector / norm
            except Exception as e:
                # Silently fail if identity update fails
                pass

        # Clear buffer after consolidation
        self.buffer = []

        return {"summary_text": summary_text, "concept_name": concept_name if memory_manager else None}

