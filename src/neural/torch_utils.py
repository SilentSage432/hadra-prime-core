# src/neural/torch_utils.py
# A131: Torch Utility Stubs (Pre-PyTorch)
"""
Torch Utility Stubs (Pre-PyTorch)

---------------------------------
This file creates placeholder interfaces for tensor operations
without importing torch yet.

Actual torch functionality will be wired in Phase A135.
"""


def safe_tensor(data):
    """
    Before PyTorch is installed, this simply returns the input.
    After A135, this will convert lists → torch tensors.
    """
    return data


def safe_norm(vector):
    """
    Compute Euclidean norm without torch.
    Replaced with torch.norm() in A135.
    """
    return sum(x * x for x in vector) ** 0.5


def safe_cosine_similarity(a, b):
    """
    Basic cosine similarity without torch.
    Replaced with torch-based version later.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = safe_norm(a)
    norm_b = safe_norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def is_tensor(obj):
    """
    Until PyTorch installs, no object is a tensor.
    """
    return False

