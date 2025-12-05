# prime-core/neural/neural_thought_selector.py

"""
Neural Thought Selection Engine (A144)

--------------------------------------

Chooses PRIME's next internal thought based on:

- salience with the attention vector

- coherence with the fusion vector

- stability vs. drift balance

- relevance to long-term identity

- novelty detection

- multi-timescale memory interaction

Inputs:

- candidate embeddings (generated from encoded text)

- cognitive fusion vector

- attention engine

- neural memory manager (episodic + semantic)

Output:

- a single chosen embedding representing the next internal thought

"""

import torch

from .torch_utils import safe_cosine_similarity, safe_tensor, is_tensor

class NeuralThoughtSelector:

    def __init__(self, novelty_weight=0.20, coherence_weight=0.50, salience_weight=0.30):

        self.novelty_weight = novelty_weight

        self.coherence_weight = coherence_weight

        self.salience_weight = salience_weight

    def score_thought(self, embedding, fusion_vec, attention_engine, memory_manager):

        """

        Computes a score for how appropriate a candidate thought is.

        """

        emb = safe_tensor(embedding)

        # 1. Salience — how relevant the thought is to PRIME's attention

        salience = attention_engine.salience(emb)

        # 2. Coherence — similarity to cognitive fusion state (self-consistency)

        coherence = safe_cosine_similarity(emb, fusion_vec) if fusion_vec is not None else 0.0

        # 3. Novelty — how different it is from recent episodic memory

        if memory_manager is not None:

            recent = memory_manager.episodic.retrieve_similar(emb, top_k=1)

            if recent and len(recent) > 0:

                novelty = 1.0 - recent[0][0]  # 1 - similarity

            else:

                novelty = 1.0

        else:

            # If memory manager not available, assume maximum novelty

            novelty = 1.0

        # Weighted total score

        score = (

            self.salience_weight * salience +

            self.coherence_weight * coherence +

            self.novelty_weight * novelty

        )

        return score, {

            "salience": salience,

            "coherence": coherence,

            "novelty": novelty,

            "total": score

        }

    def select(self, candidate_embeddings, fusion_vec, attention_engine, memory_manager):

        """

        Choose the best thought among candidates.

        """

        if not candidate_embeddings:

            return None, None

        best = None

        best_score = -999

        best_debug = None

        for emb in candidate_embeddings:

            score, dbg = self.score_thought(emb, fusion_vec, attention_engine, memory_manager)

            if score > best_score:

                best_score = score

                best = emb

                best_debug = dbg

        return best, best_debug

