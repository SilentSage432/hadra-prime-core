# src/cognition/emergent/concept_types.py

from dataclasses import dataclass
from typing import List
import time
import uuid
import torch

@dataclass
class EmergentConcept:
    id: str
    centroid: torch.Tensor
    variance: float
    occurrences: int
    first_seen: float
    last_seen: float
    stability_score: float
    decay_score: float
    last_updated: float
    maturity: float
    quiet: bool
    tension: float
    conflicts: List[str]

    @staticmethod
    def create(initial_vector: torch.Tensor):
        now = time.time()
        return EmergentConcept(
            id=str(uuid.uuid4()),
            centroid=initial_vector.clone(),
            variance=0.0,
            occurrences=1,
            first_seen=now,
            last_seen=now,
            stability_score=0.0,
            decay_score=0.0,
            last_updated=now,
            maturity=0.0,
            quiet=True,
            tension=0.0,
            conflicts=[],
        )

