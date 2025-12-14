# src/cognition/emergent/concept_observer.py

import torch
import torch.nn.functional as F
import time
from .concept_store import ConceptStore
from .concept_types import EmergentConcept

class ConceptObserver:
    def __init__(
        self,
        distance_threshold: float = 0.15,
        stability_gain: float = 0.05,
        maturity_threshold: float = 0.35,
        dominance_threshold: float = 0.65,
        tension_similarity_threshold: float = 0.45,
        tension_increment: float = 0.05,
        tension_decay: float = 0.01
    ):
        self.store = ConceptStore()
        self.distance_threshold = distance_threshold
        self.stability_gain = stability_gain
        self.maturity_threshold = maturity_threshold
        self.dominance_threshold = dominance_threshold
        self.tension_similarity_threshold = tension_similarity_threshold
        self.tension_increment = tension_increment
        self.tension_decay = tension_decay

    def observe(self, fusion_vector: torch.Tensor):
        """
        Observe a fusion vector and update or create concepts.
        Silent failure - never affects runtime.
        """
        try:
            if fusion_vector is None:
                return
            
            current_time = time.time()
            
            # Decay pass - concepts weaken if not re-observed
            for concept in self.store.concepts:
                time_delta = current_time - concept.last_seen
                decay_amount = min(time_delta * 0.00001, 0.05)
                
                concept.decay_score += decay_amount
                concept.stability_score = max(
                    0.0,
                    concept.stability_score - decay_amount
                )
                # Tension decay over time
                concept.tension = max(0.0, concept.tension - self.tension_decay)
                if concept.tension == 0.0:
                    concept.conflicts.clear()
            
            # Collect all matching concepts (active in this observation)
            active_concepts = []
            matched = False

            for concept in self.store.concepts:
                dist = torch.norm(fusion_vector - concept.centroid).item()
                if dist <= self.distance_threshold:
                    concept.occurrences += 1
                    concept.last_seen = current_time
                    concept.last_updated = current_time

                    # Drift-aware centroid update (momentum smoothing)
                    alpha = 0.15  # learning rate
                    concept.centroid = (
                        (1 - alpha) * concept.centroid +
                        alpha * fusion_vector
                    )

                    concept.variance = dist
                    concept.stability_score = min(1.0, concept.stability_score + self.stability_gain)
                    # Maturity accumulation - earned slowly through reoccurrence
                    concept.maturity += 0.01
                    concept.maturity = min(concept.maturity, 1.0)
                    active_concepts.append(concept)
                    matched = True

            # Detect tension between co-activating concepts
            if len(active_concepts) > 1:
                for i, a in enumerate(active_concepts):
                    for b in active_concepts[i+1:]:
                        if a.id == b.id:
                            continue
                        
                        # Both must have non-zero maturity
                        if a.maturity == 0.0 or b.maturity == 0.0:
                            continue
                        
                        # Compute cosine similarity
                        try:
                            # Ensure vectors are 1D and have same shape
                            vec_a = a.centroid.flatten()
                            vec_b = b.centroid.flatten()
                            if vec_a.shape == vec_b.shape:
                                sim = F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0), dim=1).item()
                                
                                if sim < self.tension_similarity_threshold:
                                    a.tension += self.tension_increment
                                    b.tension += self.tension_increment
                                    a.tension = min(a.tension, 1.0)
                                    b.tension = min(b.tension, 1.0)
                                    
                                    if b.id not in a.conflicts:
                                        a.conflicts.append(b.id)
                                    if a.id not in b.conflicts:
                                        b.conflicts.append(a.id)
                        except Exception:
                            # Silent failure - skip tension detection if error
                            pass

            if not matched:
                self.store.concepts.append(
                    EmergentConcept.create(fusion_vector)
                )

            # Consolidation pass - merge highly similar concepts
            self.consolidate()
            
            # Classification pass - quiet vs dominant concepts
            self._classify_concepts()
            
            self.store.save()
        except Exception:
            # ECFL cannot stop ADRAE - silent failure
            pass

    def consolidate(self):
        """
        Merge highly similar concepts to prevent redundancy.
        Silent failure - never affects runtime.
        """
        try:
            merged = []
            used = set()

            for i, c1 in enumerate(self.store.concepts):
                if i in used:
                    continue

                for j, c2 in enumerate(self.store.concepts):
                    if i == j or j in used:
                        continue

                    dist = torch.norm(c1.centroid - c2.centroid).item()
                    if dist < self.distance_threshold * 0.75:
                        # Merge c2 into c1
                        c1.centroid = (c1.centroid + c2.centroid) / 2
                        c1.occurrences += c2.occurrences
                        c1.stability_score += c2.stability_score * 0.5
                        # Preserve earliest first_seen
                        c1.first_seen = min(c1.first_seen, c2.first_seen)
                        # Preserve latest last_seen
                        c1.last_seen = max(c1.last_seen, c2.last_seen)
                        # Combine decay scores
                        c1.decay_score = max(c1.decay_score, c2.decay_score)
                        # Preserve higher maturity when merging
                        c1.maturity = max(c1.maturity, c2.maturity)
                        # Preserve higher tension when merging
                        c1.tension = max(c1.tension, c2.tension)
                        # Merge conflicts lists
                        for conflict_id in c2.conflicts:
                            if conflict_id not in c1.conflicts and conflict_id != c1.id:
                                c1.conflicts.append(conflict_id)
                        # Quiet status will be recalculated in classification pass
                        used.add(j)

                merged.append(c1)
                used.add(i)

            self.store.concepts = merged
        except Exception:
            # Silent failure - consolidation must not affect runtime
            pass

    def _classify_concepts(self):
        """
        Classify concepts as quiet or dominant based on maturity.
        Silent failure - never affects runtime.
        """
        try:
            for concept in self.store.concepts:
                if concept.maturity < self.maturity_threshold:
                    concept.quiet = True
                elif concept.maturity >= self.dominance_threshold:
                    concept.quiet = False
        except Exception:
            # Silent failure - classification must not affect runtime
            pass

    def has_dominant_concepts(self):
        """
        Returns True if any concept is dominant (not quiet).
        Used by translator layer, UI projection, and response gating.
        """
        try:
            return any(not c.quiet for c in self.store.concepts)
        except Exception:
            # Silent failure - return False on error
            return False

    def has_unresolved_tension(self):
        """
        Returns True if any concept has unresolved tension (tension > 0.2).
        Used to gate expression, UI cues, and translator ambiguity markers.
        """
        try:
            return any(c.tension > 0.2 for c in self.store.concepts)
        except Exception:
            # Silent failure - return False on error
            return False

