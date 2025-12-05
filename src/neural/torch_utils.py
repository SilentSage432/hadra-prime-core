# prime-core/neural/torch_utils.py

"""
Torch Utilities (Activated in A136)

-----------------------------------

This module now uses real torch tensor operations.

If torch is not available, it gracefully falls back to Python lists.

"""

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    F = None

def safe_tensor(data):

    """

    Convert Python lists into torch tensors.

    If it's already a tensor, returns it unchanged.

    """

    if not TORCH_AVAILABLE:

        return data

    if isinstance(data, torch.Tensor):

        return data

    try:

        return torch.tensor(data, dtype=torch.float32)

    except Exception:

        return data

def safe_norm(vector):

    """

    Compute norm using torch if possible.

    """

    if not TORCH_AVAILABLE:

        return sum(x * x for x in vector) ** 0.5

    tensor = safe_tensor(vector)

    if isinstance(tensor, torch.Tensor):

        return torch.linalg.norm(tensor).item()

    return sum(x * x for x in vector) ** 0.5

def safe_cosine_similarity(a, b):

    """

    Torch cosine similarity fallback wrapper.

    """

    if not TORCH_AVAILABLE:

        # Fallback to manual calculation

        dot = sum(x * y for x, y in zip(a, b))

        norm_a = safe_norm(a)

        norm_b = safe_norm(b)

        if norm_a == 0 or norm_b == 0:

            return 0.0

        return dot / (norm_a * norm_b)

    t1, t2 = safe_tensor(a), safe_tensor(b)

    if isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):

        # Torch expects shape [1, dim] to compute similarity

        t1 = t1.unsqueeze(0)

        t2 = t2.unsqueeze(0)

        return F.cosine_similarity(t1, t2).item()

    # Fallback (should rarely happen now)

    dot = sum(x * y for x, y in zip(a, b))

    norm_a = safe_norm(a)

    norm_b = safe_norm(b)

    if norm_a == 0 or norm_b == 0:

        return 0.0

    return dot / (norm_a * norm_b)

def is_tensor(obj):

    if not TORCH_AVAILABLE:

        return False

    return isinstance(obj, torch.Tensor)
