# prime-core/neural/dual_mind_sync.py

"""
Dual-Mind Neural Sync Layer (A140)

----------------------------------

Establishes neural-level synchronization between PRIME and SAGE.

This layer does NOT require SAGE to be running yet.

It prepares vectors, alignment signals, and shared conceptual space.

When SAGE is online, the same structure will sync back.

"""

import torch

from .torch_utils import safe_tensor, safe_cosine_similarity

class DualMindSync:

    def __init__(self):

        # PRIME's side of shared embeddings

        self.prime_intent_vector = None

        self.prime_concept_vector = None

        # Placeholder for SAGE-side (can be sent/received via gateway)

        self.sage_intent_vector = None

        self.sage_concept_vector = None

    def update_prime_vectors(self, identity_vec, concept_vec):

        """

        Called by PRIME to provide its neural anchors.

        """

        if identity_vec is not None:

            self.prime_intent_vector = safe_tensor(identity_vec)

        if concept_vec is not None:

            self.prime_concept_vector = safe_tensor(concept_vec)

    def ingest_sage_vectors(self, intent_vec, concept_vec):

        """

        Used later when SAGE sends neural embeddings back.

        """

        if intent_vec is not None:

            self.sage_intent_vector = safe_tensor(intent_vec)

        if concept_vec is not None:

            self.sage_concept_vector = safe_tensor(concept_vec)

    def coherence_score(self):

        """

        Measures neural coherence between the two minds.

        """

        if self.prime_intent_vector is None or self.sage_intent_vector is None:

            return None

        return safe_cosine_similarity(

            self.prime_intent_vector, self.sage_intent_vector

        )

    def concept_alignment(self):

        """

        Measures semantic space alignment

        between PRIME and SAGE conceptual vectors.

        """

        if self.prime_concept_vector is None or self.sage_concept_vector is None:

            return None

        return safe_cosine_similarity(

            self.prime_concept_vector, self.sage_concept_vector

        )

    def status(self):

        return {

            "intent_alignment": self.coherence_score(),

            "concept_alignment": self.concept_alignment(),

            "prime_intent_dim": self.prime_intent_vector.numel() if self.prime_intent_vector is not None else None,

            "sage_intent_dim": self.sage_intent_vector.numel() if self.sage_intent_vector is not None else None,

        }

