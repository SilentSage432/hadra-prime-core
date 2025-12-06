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

from .torch_utils import safe_cosine_similarity, safe_tensor, is_tensor, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch

class NeuralThoughtSelector:

    def __init__(self, novelty_weight=0.20, coherence_weight=0.50, salience_weight=0.30):

        self.novelty_weight = novelty_weight

        self.coherence_weight = coherence_weight

        self.salience_weight = salience_weight

    def score_thought(self, embedding, fusion_vec, attention_engine, memory_manager, skill_bias=0.0):

        """

        Computes a score for how appropriate a candidate thought is.
        
        A217 — Now includes skill bias influence.

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

        # A217 — Skill bias (weighted influence from skill vectors)
        skill_weight = 0.15  # 15% weight for skill influence

        # Weighted total score

        score = (

            self.salience_weight * salience +

            self.coherence_weight * coherence +

            self.novelty_weight * novelty +

            skill_weight * skill_bias

        )

        return score, {

            "salience": salience,

            "coherence": coherence,

            "novelty": novelty,

            "skill_bias": skill_bias,

            "total": score

        }

    def select(self, candidate_embeddings, fusion_vec, attention_engine, memory_manager, goal_modulation=None, competency_bias=0.0):

        """

        Choose the best thought among candidates.

        A204 — Now includes goal modulation influence on scoring.
        A215 — Now includes cross-domain skill transfer.
        A217 — Now includes skill bias.
        A218 — Now includes competency bias.
        A219 — Now includes competency activation bias.

        """

        # A215 — Cross-domain transfer shaping
        # Add transferred skills as additional candidates
        shaped_candidates = list(candidate_embeddings) if candidate_embeddings else []
        
        if hasattr(attention_engine, 'bridge') and hasattr(attention_engine.bridge, 'skill_generalizer'):
            try:
                # Get current focus vector (attention or fusion)
                focus = None
                if hasattr(attention_engine, 'last_focus_vector') and attention_engine.last_focus_vector is not None:
                    focus = attention_engine.last_focus_vector
                elif fusion_vec is not None:
                    focus = fusion_vec
                
                if focus is not None:
                    # Find similar patterns and transfer them
                    similar_patterns = attention_engine.bridge.skill_generalizer.find_similar_patterns(
                        focus,
                        top_k=2
                    )
                    
                    for pattern_entry in similar_patterns:
                        pattern = pattern_entry.get("pattern")
                        if pattern is not None:
                            # Transfer skill pattern to current context
                            transferred = attention_engine.bridge.skill_generalizer.transfer_skill(
                                pattern,
                                focus
                            )
                            if transferred is not None:
                                shaped_candidates.append(transferred)
            except Exception as e:
                # If transfer fails, continue without it
                pass

        if not shaped_candidates:

            return None, None

        # A217 — Apply skill priors to thought selection
        skill_vectors = []
        if hasattr(attention_engine, 'bridge') and hasattr(attention_engine.bridge, 'skills'):
            skill_vectors = attention_engine.bridge.skills.get_all_skill_vectors()
        
        # A218 — Get competency centroids for biasing
        competency_centroids = []
        if hasattr(attention_engine, 'bridge') and hasattr(attention_engine.bridge, 'competencies'):
            competency_centroids = attention_engine.bridge.competencies.get_centroids()
        
        best = None

        best_score = -999

        best_debug = None

        for emb in shaped_candidates:
            # A217 — Compute skill bias for this candidate
            skill_bias = 0.0
            if skill_vectors and fusion_vec is not None:
                # Mean similarity to all skill vectors
                sims = []
                for sv in skill_vectors:
                    sim = safe_cosine_similarity(emb, sv)
                    if sim is not None:
                        sims.append(sim)
                if sims:
                    skill_bias = sum(sims) / len(sims)
            
            # A218 — Compute competency bias for this candidate
            comp_bias = 0.0
            if competency_centroids and fusion_vec is not None:
                # Mean similarity to all competency centroids
                comp_sims = []
                for centroid in competency_centroids:
                    if centroid is not None:
                        sim = safe_cosine_similarity(emb, centroid)
                        if sim is not None:
                            comp_sims.append(sim)
                if comp_sims:
                    comp_bias = sum(comp_sims) / len(comp_sims)
            
            # Combine skill and competency bias (competency has slightly higher weight)
            combined_bias = (skill_bias * 0.4) + (comp_bias * 0.6)
            
            # A219 — Add direct competency activation bias
            # This is the bias from activated competencies (separate from centroid similarity)
            final_bias = combined_bias + (competency_bias * 0.3)  # 30% weight for activation bias
            
            score, dbg = self.score_thought(emb, fusion_vec, attention_engine, memory_manager, skill_bias=final_bias)
            
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

