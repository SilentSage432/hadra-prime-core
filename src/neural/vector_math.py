# src/neural/vector_math.py
# A131: Vector Math Utilities
"""
Vector Math Utilities

---------------------
These functions operate on plain Python lists for now.
Upgraded to torch operations in A135–A138.
"""


def normalize(vector):
    try:
        from .torch_utils import safe_norm
    except ImportError:
        from torch_utils import safe_norm
    n = safe_norm(vector)
    return [v / n for v in vector] if n > 0 else vector


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def magnitude(vec):
    try:
        from .torch_utils import safe_norm
    except ImportError:
        from torch_utils import safe_norm
    return safe_norm(vec)


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def subtract(a, b):
    return [x - y for x, y in zip(a, b)]


def scale(vector, factor):
    return [x * factor for x in vector]

