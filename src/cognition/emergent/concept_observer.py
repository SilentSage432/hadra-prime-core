# src/cognition/emergent/concept_observer.py

import torch
import time
from .concept_store import ConceptStore
from .concept_types import EmergentConcept

class ConceptObserver:
    def __init__(
        self,
        distance_threshold: float = 0.15,
        stability_gain: float = 0.05
    ):
        self.store = ConceptStore()
        self.distance_threshold = distance_threshold
        self.stability_gain = stability_gain

    def observe(self, fusion_vector: torch.Tensor):
        """
        Observe a fusion vector and update or create concepts.
        Silent failure - never affects runtime.
        """
        try:
            if fusion_vector is None:
                return
            
            matched = False

            for concept in self.store.concepts:
                dist = torch.norm(fusion_vector - concept.centroid).item()
                if dist <= self.distance_threshold:
                    concept.occurrences += 1
                    concept.last_seen = time.time()

                    # Update centroid (running average)
                    concept.centroid = (
                        concept.centroid * (concept.occurrences - 1) +
                        fusion_vector
                    ) / concept.occurrences

                    concept.variance = dist
                    concept.stability_score = min(1.0, concept.stability_score + self.stability_gain)
                    matched = True
                    break

            if not matched:
                self.store.concepts.append(
                    EmergentConcept.create(fusion_vector)
                )

            self.store.save()
        except Exception:
            # ECFL cannot stop ADRAE - silent failure
            pass

