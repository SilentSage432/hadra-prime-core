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

    def select(self, candidate_embeddings, fusion_vec, attention_engine, memory_manager, goal_modulation=None):

        """

        Choose the best thought among candidates.

        A204 — Now includes goal modulation influence on scoring.

        """

        if not candidate_embeddings:

            return None, None

        best = None

        best_score = -999

        best_debug = None

        for emb in candidate_embeddings:

            score, dbg = self.score_thought(emb, fusion_vec, attention_engine, memory_manager)
            
            # === A204: Inject goal modulation influence ===
            if goal_modulation is not None:
                emb_tensor = safe_tensor(emb)
                mod_tensor = safe_tensor(goal_modulation)
                
                if TORCH_AVAILABLE and isinstance(emb_tensor, torch.Tensor) and isinstance(mod_tensor, torch.Tensor):
                    # Ensure same dimensions
                    if emb_tensor.shape == mod_tensor.shape:
                        # Add modulation influence (20% weight)
                        mod_influence = torch.dot(emb_tensor.flatten(), mod_tensor.flatten()).item() * 0.2
                        score += mod_influence
                elif not TORCH_AVAILABLE:
                    # Python list fallback
                    if hasattr(emb_tensor, '__iter__') and hasattr(mod_tensor, '__iter__'):
                        emb_list = list(emb_tensor) if not isinstance(emb_tensor, list) else emb_tensor
                        mod_list = list(mod_tensor) if not isinstance(mod_tensor, list) else mod_tensor
                        if len(emb_list) == len(mod_list):
                            mod_influence = sum(e * m for e, m in zip(emb_list, mod_list)) * 0.2
                            score += mod_influence

            if score > best_score:

                best_score = score

                best = emb

                best_debug = dbg

        # Add goal modulation info to debug output
        if best_debug is not None:
            best_debug["goal_modulation_applied"] = goal_modulation is not None

        return best, best_debug

