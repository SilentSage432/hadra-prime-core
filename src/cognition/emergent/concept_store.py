# src/cognition/emergent/concept_store.py

import os
import json
import time
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
                # Handle migration from old format (backward compatibility)
                decay_score = c.get("decay_score", 0.0)
                last_updated = c.get("last_updated", c.get("last_seen", time.time()))
                maturity = c.get("maturity", 0.0)
                quiet = c.get("quiet", True)  # Default to quiet for old concepts
                
                self.concepts.append(
                    EmergentConcept(
                        id=c["id"],
                        centroid=torch.tensor(c["centroid"]),
                        variance=c["variance"],
                        occurrences=c["occurrences"],
                        first_seen=c["first_seen"],
                        last_seen=c["last_seen"],
                        stability_score=c["stability_score"],
                        decay_score=decay_score,
                        last_updated=last_updated,
                        maturity=maturity,
                        quiet=quiet,
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
                        "decay_score": c.decay_score,
                        "last_updated": c.last_updated,
                        "maturity": c.maturity,
                        "quiet": c.quiet,
                    }
                    for c in self.concepts
                ], f, indent=2)
        except Exception:
            # Silent failure - observer must not affect runtime
            pass

