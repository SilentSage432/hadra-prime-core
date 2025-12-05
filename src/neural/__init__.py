# src/neural/__init__.py
# A131: Neural utilities package initialization

"""
Neural Utilities Package

This package provides pre-PyTorch utilities for tensor operations,
vector math, and neural pipeline management.

All utilities are torch-free and safe to use before PyTorch installation.
"""

from .torch_utils import (
    safe_tensor,
    safe_norm,
    safe_cosine_similarity,
    is_tensor
)

from .vector_math import (
    normalize,
    dot,
    magnitude,
    add,
    subtract,
    scale
)

from .tensor_pipeline import TensorPipeline

from .neural_constants import (
    DEFAULT_VECTOR_SIZE,
    MIN_CONFIDENCE,
    MAX_CONFIDENCE,
    NEURAL_DEBUG
)

__all__ = [
    # torch_utils
    'safe_tensor',
    'safe_norm',
    'safe_cosine_similarity',
    'is_tensor',
    # vector_math
    'normalize',
    'dot',
    'magnitude',
    'add',
    'subtract',
    'scale',
    # tensor_pipeline
    'TensorPipeline',
    # neural_constants
    'DEFAULT_VECTOR_SIZE',
    'MIN_CONFIDENCE',
    'MAX_CONFIDENCE',
    'NEURAL_DEBUG',
]

