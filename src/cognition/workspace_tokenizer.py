# prime-core/cognition/workspace_tokenizer.py

"""
A174 — Conscious Workspace Tokenization & Temporal Binding
----------------------------------------------------------
Converts continuous workspace embeddings into proto-tokens
and binds them across time to form structured internal sequences.
"""

import math
from collections import deque

try:
    import torch
except ImportError:
    torch = None

from ..neural.torch_utils import safe_tensor, TORCH_AVAILABLE


class WorkspaceTokenizer:
    """
    A174 — Converts continuous workspace embeddings into proto-tokens
    and binds them across time to form structured internal sequences.
    """

    def __init__(self, max_sequence=12):
        self.history = deque(maxlen=max_sequence)
        self.last_tokens = None

    def quantize(self, vector, num_tokens=6):
        """
        Convert a 128-dim embedding into N token-like units by chunking
        and normalizing. This is not linguistic tokenization — it is
        cognitive segmentation.
        """
        if vector is None:
            return None

        try:
            # Convert to tensor if needed
            v = safe_tensor(vector)
            if v is None:
                return None

            # Handle both torch tensors and Python lists
            if TORCH_AVAILABLE and isinstance(v, torch.Tensor):
                v = v.clone().detach()
                vector_len = v.shape[0] if len(v.shape) > 0 else len(v)
            else:
                if isinstance(v, list):
                    vector_len = len(v)
                else:
                    vector_len = len(v) if hasattr(v, '__len__') else 0
                    if vector_len == 0:
                        return None

            chunk_size = math.floor(vector_len / num_tokens)
            if chunk_size == 0:
                chunk_size = 1

            tokens = []

            for i in range(num_tokens):
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size if i < num_tokens - 1 else vector_len
                
                if TORCH_AVAILABLE and isinstance(v, torch.Tensor):
                    chunk = v[start_idx:end_idx]
                    tokens.append(chunk.mean().item())
                else:
                    # Python fallback
                    if isinstance(v, list):
                        chunk = v[start_idx:end_idx]
                    else:
                        chunk = list(v)[start_idx:end_idx]
                    if chunk:
                        tokens.append(sum(chunk) / len(chunk))
                    else:
                        tokens.append(0.0)

            return tokens
        except Exception:
            return None

    def bind_temporally(self, current_tokens):
        """
        Create a temporal binding structure linking previous and current tokens.
        """
        if current_tokens is None:
            return None

        if self.last_tokens is None:
            binding = {
                "current": current_tokens,
                "previous": None,
                "link_strength": 1.0
            }
        else:
            # link strength = similarity measure (dot product normalized)
            try:
                if len(current_tokens) == len(self.last_tokens):
                    dot = sum(a * b for a, b in zip(current_tokens, self.last_tokens))
                    # Normalize by token count for stability
                    norm_a = sum(a * a for a in current_tokens) ** 0.5
                    norm_b = sum(b * b for b in self.last_tokens) ** 0.5
                    if norm_a > 0 and norm_b > 0:
                        link_strength = dot / (norm_a * norm_b)
                    else:
                        link_strength = 0.0
                else:
                    link_strength = 0.0
            except Exception:
                link_strength = 0.0

            binding = {
                "current": current_tokens,
                "previous": self.last_tokens,
                "link_strength": float(link_strength)
            }

        self.last_tokens = current_tokens
        self.history.append(binding)
        return binding

    def get_sequence(self):
        """Get the current sequence history."""
        return list(self.history)

