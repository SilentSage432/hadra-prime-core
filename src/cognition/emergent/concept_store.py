# src/cognition/emergent/concept_store.py

import os
import json
import torch
from typing import List
from .concept_types import EmergentConcept

DATA_ROOT = "/data/substrate"
CONCEPT_PATH = os.path.join(DATA_ROOT, "concepts.json")

class ConceptStore:
    def __init__(self):
        os.makedirs(DATA_ROOT, exist_ok=True)
        self.concepts: List[EmergentConcept] = []
        self._load()

    def _load(self):
        if not os.path.exists(CONCEPT_PATH):
            return
        try:
            with open(CONCEPT_PATH, "r") as f:
                raw = json.load(f)
            for c in raw:
                self.concepts.append(
                    EmergentConcept(
                        id=c["id"],
                        centroid=torch.tensor(c["centroid"]),
                        variance=c["variance"],
                        occurrences=c["occurrences"],
                        first_seen=c["first_seen"],
                        last_seen=c["last_seen"],
                        stability_score=c["stability_score"],
                    )
                )
        except Exception:
            # Silent failure - start fresh if corrupted
            pass

    def save(self):
        try:
            with open(CONCEPT_PATH, "w") as f:
                json.dump([
                    {
                        "id": c.id,
                        "centroid": c.centroid.tolist(),
                        "variance": c.variance,
                        "occurrences": c.occurrences,
                        "first_seen": c.first_seen,
                        "last_seen": c.last_seen,
                        "stability_score": c.stability_score,
                    }
                    for c in self.concepts
                ], f, indent=2)
        except Exception:
            # Silent failure - observer must not affect runtime
            pass

