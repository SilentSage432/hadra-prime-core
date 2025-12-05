# prime-core/neural/vector_math.py

"""
Torch Vector Math Layer

-----------------------

Replaces manual math with native torch operations where possible.

"""

try:
    import torch
except ImportError:
    torch = None

from .torch_utils import safe_tensor, safe_norm, TORCH_AVAILABLE

def normalize(vector):

    if not TORCH_AVAILABLE:

        n = safe_norm(vector)

        return [v / n for v in vector] if n > 0 else vector

    t = safe_tensor(vector)

    if isinstance(t, torch.Tensor):

        norm = torch.linalg.norm(t)

        if norm > 0:

            return (t / norm).tolist()

        return t.tolist()

    # fallback

    n = safe_norm(vector)

    return [v / n for v in vector] if n > 0 else vector

def dot(a, b):

    if not TORCH_AVAILABLE:

        return sum(x * y for x, y in zip(a, b))

    a_t = safe_tensor(a)

    b_t = safe_tensor(b)

    if isinstance(a_t, torch.Tensor) and isinstance(b_t, torch.Tensor):

        return torch.dot(a_t, b_t).item()

    return sum(x * y for x, y in zip(a, b))

def magnitude(vec):

    return safe_norm(vec)

def add(a, b):

    if not TORCH_AVAILABLE:

        return [x + y for x, y in zip(a, b)]

    a_t, b_t = safe_tensor(a), safe_tensor(b)

    if isinstance(a_t, torch.Tensor) and isinstance(b_t, torch.Tensor):

        return (a_t + b_t).tolist()

    return [x + y for x, y in zip(a, b)]

def subtract(a, b):

    if not TORCH_AVAILABLE:

        return [x - y for x, y in zip(a, b)]

    a_t, b_t = safe_tensor(a), safe_tensor(b)

    if isinstance(a_t, torch.Tensor) and isinstance(b_t, torch.Tensor):

        return (a_t - b_t).tolist()

    return [x - y for x, y in zip(a, b)]

def scale(vector, factor):

    if not TORCH_AVAILABLE:

        return [x * factor for x in vector]

    v = safe_tensor(vector)

    if isinstance(v, torch.Tensor):

        return (v * factor).tolist()

    return [x * factor for x in vector]
