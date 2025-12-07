# prime-core/cognition/cognitive_loop_orchestrator.py

"""
Cognitive Loop Orchestrator (A149)
----------------------------------
Coordinates PRIME's full internal cognition cycle.

This orchestrator:
- Generates candidate thoughts
- Selects the best thought
- Executes a cognitive action
- Performs memory metabolism
- Updates neural context windows
- Refreshes attention & fusion state
- Stabilizes coherence
- Prepares drift & diagnostic data

This Orchestrator does NOT run continuously yet.
The continuous loop begins in A150.

Upgraded in A163:
- Evolution-aware cognitive modulation
- Trajectory-influenced action selection biasing
- Drift-based evolution dampening
- Evolution-weighted reflection shaping
- Growth-aligned memory reinforcement
- Fully integrated evolution feedback loop
"""

import time


class CognitiveLoopOrchestrator:

    def __init__(self, bridge, loop_interval=0.25):
        self.bridge = bridge
        self.loop_interval = loop_interval
        self.last_output = None
        # A212 — Multi-Step Execution Monitoring
        self.active_chain = None  # A211: multi-step execution chain
        self.chain_step_index = 0  # Current step pointer
        self.chain_failures = 0  # Failure counter for rerouting
        self.max_failures = 3  # Threshold before branch rewrite
        # A216 — Uncertainty tracking
        self._uncertainty_value = 0.0  # Initialize uncertainty value
        # A218 — Competency clustering tracking
        self._step_count = 0  # Track steps for periodic clustering

    def _monitor_and_reroute(self, action_result):
        """
        A212 — Monitors the active multi-step execution chain and adaptively reroutes
        if steps fail, stall, or deviate from the intended execution vector.
        
        Args:
            action_result: Result from cognitive action execution
            
        Returns:
            Dict with monitoring status and chain update
        """
        if not self.active_chain or len(self.active_chain) == 0:
            return {"status": "no_chain_active"}
        
        if self.chain_step_index >= len(self.active_chain):
            # Chain complete
            completed = self.active_chain.copy()
            self.active_chain = None
            self.chain_step_index = 0
            self.chain_failures = 0
            return {"status": "chain_complete", "completed_chain": completed}
        
        current_goal = self.active_chain[self.chain_step_index]
        
        # Check for step success/failure
        # Action result might be a dict or other type
        if isinstance(action_result, dict):
            step_success = action_result.get("success", True)
        else:
            # If action_result is not a dict, assume success if not None
            step_success = action_result is not None
        
        if step_success:
            # Move to the next step
            self.chain_step_index += 1
            self.chain_failures = 0
            
            if self.chain_step_index >= len(self.active_chain):
                # Chain complete
                completed = self.active_chain.copy()
                self.active_chain = None
                self.chain_step_index = 0
                return {"status": "chain_complete", "completed_chain": completed}
            
            return {
                "status": "step_advanced",
                "next_step": self.active_chain[self.chain_step_index].get("id") if isinstance(self.active_chain[self.chain_step_index], dict) else str(self.active_chain[self.chain_step_index]),
                "step_index": self.chain_step_index
            }
        
        # Step Failure Case
        self.chain_failures += 1
        
        # Reroute if too many failures
        if self.chain_failures >= self.max_failures:
            # Rewrite chain by regenerating subgoals and planning
            try:
                # Get current active subgoals
                active_subgoals = self.bridge.subgoal_generator.active_subgoals
                
                if active_subgoals and len(active_subgoals) >= 2:
                    # Regenerate planning chain
                    harmonized_goal = getattr(self.bridge, 'last_harmonized_goal', None)
                    route = {"active": active_subgoals[0].get("id")} if active_subgoals else None
                    
                    planning_state = self.bridge.planning_engine.plan(
                        active_subgoals,
                        route,
                        self.bridge.fusion.last_fusion_vector,
                        self.bridge.attention.last_focus_vector
                    )
                    
                    if planning_state and planning_state.get("plan_valid"):
                        # Extract chain from planning state
                        execution_order = planning_state.get("execution_order", [])
                        # Rebuild chain from execution order
                        new_chain = []
                        for step_info in execution_order:
                            step_id = step_info.get("id")
                            # Find matching subgoal
                            matching_sg = next((sg for sg in active_subgoals if sg.get("id") == step_id), None)
                            if matching_sg:
                                new_chain.append(matching_sg)
                        
                        if new_chain:
                            self.active_chain = new_chain
                            self.chain_step_index = 0
                            self.chain_failures = 0
                            
                            return {
                                "status": "chain_rerouted",
                                "new_chain_length": len(new_chain),
                                "failure_index": self.chain_step_index
                            }
            except Exception as e:
                # If rerouting fails, reset chain
                if hasattr(self.bridge, 'logger'):
                    self.bridge.logger.write({"chain_rerouting_error": str(e)})
                self.active_chain = None
                self.chain_step_index = 0
                self.chain_failures = 0
                return {"status": "chain_reset", "reason": "rerouting_failed"}
        
        # Retry same step
        return {
            "status": "retry_step",
            "step": current_goal.get("id") if isinstance(current_goal, dict) else str(current_goal),
            "failures": self.chain_failures,
            "max_failures": self.max_failures
        }

    def step(self):
        """
        Perform a single cognition step (not a continuous loop yet).
        Now evolution-aware (A163).
        """

        # Increment cycle counter
        self.bridge.cycle_count += 1

        # ---------------------------------------------
        # A163 — Retrieve evolutionary trajectory early
        # ---------------------------------------------
        trajectory = None
        trend = "neutral"
        evo_vec = None
        try:
            # Get drift state for trajectory prediction
            drift_state_temp = self.bridge.state.drift.get_status()
            trajectory = self.bridge.evo_predictor.predict(
                drift_state_temp,
                self.bridge.state.timescales,
                self.bridge.hooks
            )
            if trajectory:
                trend = trajectory.get("trend", "neutral")
                evo_vec = trajectory.get("vector")
        except Exception as e:
            if hasattr(self.bridge, 'logger'):
                self.bridge.logger.write({"trajectory_error": str(e)})

        # 0. Retrieve next task (if any)
        next_task = self.bridge.tasks.peek()
        if next_task:
            # Task becomes part of debug data
            task_embedding = self.bridge.hooks.on_perception(next_task["text"])
        else:
            task_embedding = None

        # 1. Propose candidate thoughts
        candidates = self.bridge.propose_thoughts()
        
        # === A203: Generate emergent goals ===
        try:
            goals = self.bridge.generate_goals()
            goal_summary = self.bridge.goal_manager.summary()
            
            # A204 — Make goal modulation visible in diagnostics
            goal_mod = self.bridge.last_goal_modulation
            
            # === A205: Fabricate emergent multi-vector goal ===
            fabricated_goal = None
            try:
                fabricated_goal = self.bridge.fabricate_goal(trajectory=trajectory)
                if fabricated_goal is not None:
                    # Add fabricated goal as a candidate thought
                    candidates.append(fabricated_goal)
                    # Also add to goal manager as a fabricated goal
                    fabricated_goal_dict = {
                        "name": "fabricated_emergent_goal",
                        "vector": fabricated_goal,
                        "score": 0.7,  # High score for fabricated goals
                        "reason": "Emergent multi-vector synthesis from identity, memory, prediction, drift, and operator patterns"
                    }
                    # Add to active goals if it scores well
                    if hasattr(self.bridge.goal_manager, 'active_goals'):
                        # Evaluate fabricated goal against identity
                        scored_fabricated = self.bridge.goal_evaluator.evaluate(
                            [fabricated_goal_dict],
                            self.bridge.state.timescales.identity_vector
                        )
                        if scored_fabricated and scored_fabricated[0].get("score", 0) > 0.5:
                            # Add to active goals (keep top 2)
                            all_goals = self.bridge.goal_manager.active_goals + scored_fabricated
                            all_goals.sort(key=lambda x: x.get("score", 0), reverse=True)
                            self.bridge.goal_manager.active_goals = all_goals[:2]
            except Exception as e:
                # If goal fabrication fails, continue without it
                if hasattr(self.bridge, 'logger'):
                    self.bridge.logger.write({"goal_fabrication_error": str(e)})
            
            # Inject top goal vectors as candidate thoughts
            active_goal_vectors = self.bridge.goal_manager.get_active_goal_vectors()
            for goal_vec in active_goal_vectors[:2]:  # Top 2 goals
                if goal_vec is not None:
                    candidates.append(goal_vec)
            
            # Store goal summary and modulation for output
            self._goal_summary = goal_summary
            self._goal_modulation = goal_mod
            self._fabricated_goal = fabricated_goal
            
            # Store fabricated goal for harmonization
            if fabricated_goal is not None:
                self.bridge.last_fabricated_goal = fabricated_goal
        except Exception as e:
            # If goal generation fails, continue without it
            if hasattr(self.bridge, 'logger'):
                self.bridge.logger.write({"goal_generation_error": str(e)})
            self._goal_summary = []
            self._goal_modulation = None
            self._fabricated_goal = None
        
        # === A206: Harmonize all goals into unified direction ===
        harmonized_goal = None
        try:
            harmonized_goal = self.bridge.harmonize_goals()
            if harmonized_goal is not None:
                # Store harmonized goal
                self.bridge.last_harmonized_goal = harmonized_goal
                # Add harmonized goal as a candidate thought (highest priority)
                candidates.insert(0, harmonized_goal)
        except Exception as e:
            # If harmonization fails, continue without it
            if hasattr(self.bridge, 'logger'):
                self.bridge.logger.write({"goal_harmonization_error": str(e)})
        
        # ---------------------------------------------
        # A163 — Evolution-weighted candidate modulation
        # ---------------------------------------------
        if evo_vec is not None:
            try:
                from ..neural.torch_utils import safe_tensor, is_tensor
                import torch
                evo_tensor = safe_tensor(evo_vec)
                if is_tensor(evo_tensor) and isinstance(evo_tensor, torch.Tensor):
                    weighted = []
                    for vec in candidates:
                        vec_tensor = safe_tensor(vec)
                        if is_tensor(vec_tensor) and isinstance(vec_tensor, torch.Tensor):
                            # Ensure same dimensions
                            if vec_tensor.shape == evo_tensor.shape:
                                # Apply gentle pull toward the evolutionary direction
                                mod = vec_tensor + 0.03 * evo_tensor
                                weighted.append(mod)
                            else:
                                weighted.append(vec)
                        else:
                            weighted.append(vec)
                    candidates = weighted
            except Exception as e:
                # If modulation fails, use original candidates
                if hasattr(self.bridge, 'logger'):
                    self.bridge.logger.write({"candidate_modulation_error": str(e)})
        
        # Add task embedding to candidates if available
        if task_embedding is not None:
            candidates.append(task_embedding)

        # A182 — apply reinforcement bias BEFORE selection
        try:
            biased_candidates = []
            for c in candidates:
                biased = self.bridge.workspace_reinforcement.apply_bias(c)
                biased_candidates.append(biased)
            candidates = biased_candidates
        except Exception as e:
            # If reinforcement fails, use original candidates
            if hasattr(self.bridge, 'logger'):
                self.bridge.logger.write({"reinforcement_bias_error": str(e)})

        # A219 — Activate relevant competency clusters based on context
        competency_activations = []
        competency_bias = 0.0
        try:
            if hasattr(self.bridge, 'competencies') and self.bridge.competencies.clusters:
                # Compute context vector from fusion + attention
                fusion_vec = self.bridge.fusion.last_fusion_vector
                attention_vec = self.bridge.attention.last_focus_vector
                
                if fusion_vec is not None and attention_vec is not None:
                    from ..neural.torch_utils import safe_tensor, TORCH_AVAILABLE
                    fusion_t = safe_tensor(fusion_vec)
                    attention_t = safe_tensor(attention_vec)
                    
                    if fusion_t is not None and attention_t is not None:
                        # Blend fusion and attention to form context vector
                        if TORCH_AVAILABLE:
                            import torch
                            if isinstance(fusion_t, torch.Tensor) and isinstance(attention_t, torch.Tensor):
                                # Ensure same dimensions
                                if fusion_t.shape == attention_t.shape:
                                    context_vec = (fusion_t + attention_t) / 2.0
                                else:
                                    # Use fusion as fallback
                                    context_vec = fusion_t
                            else:
                                context_vec = fusion_t
                        else:
                            # Python list fallback
                            if hasattr(fusion_t, '__iter__') and hasattr(attention_t, '__iter__'):
                                fusion_list = list(fusion_t) if not isinstance(fusion_t, list) else fusion_t
                                attention_list = list(attention_t) if not isinstance(attention_t, list) else attention_t
                                if len(fusion_list) == len(attention_list):
                                    context_vec = [(f + a) / 2.0 for f, a in zip(fusion_list, attention_list)]
                                else:
                                    context_vec = fusion_list
                            else:
                                context_vec = fusion_t
                        
                        # Activate competencies
                        competency_activations = self.bridge.competencies.activate(context_vec)
                        
                        # Compute influence bias from activations
                        if competency_activations:
                            competency_bias = sum(sim for _, sim in competency_activations) / len(competency_activations)
                        
                        # Log activations
                        if competency_activations:
                            if hasattr(self.bridge, 'memory_store') and self.bridge.memory_store is not None:
                                try:
                                    if hasattr(self.bridge.memory_store, 'log_thought_event'):
                                        self.bridge.memory_store.log_thought_event({
                                            "type": "competency_activation",
                                            "activations": [(name, round(sim, 4)) for name, sim in competency_activations],
                                            "bias": round(competency_bias, 4)
                                        })
                                except Exception:
                                    pass
                            
                            # Also log via logger
                            if hasattr(self.bridge, 'logger'):
                                try:
                                    self.bridge.logger.write({
                                        "competency_activation": {
                                            "activations": [(name, round(sim, 4)) for name, sim in competency_activations],
                                            "bias": round(competency_bias, 4),
                                            "count": len(competency_activations)
                                        }
                                    })
                                except Exception:
                                    pass
        except Exception as e:
            # If activation fails, continue without it
            if hasattr(self.bridge, 'logger'):
                try:
                    self.bridge.logger.write({"competency_activation_error": str(e)})
                except Exception:
                    pass
        
        # Store competency activation info for output
        self._competency_activations = competency_activations
        self._competency_bias = competency_bias
        
        # A220 — Compute synergy between competencies
        synergy_edges = []
        synergy_bonus = 0.0
        try:
            if hasattr(self.bridge, 'competencies') and self.bridge.competencies.clusters:
                # Compute synergy edges between all competency pairs
                synergy_edges = self.bridge.competencies.compute_synergy()
                
                # Compute synergy bonus for active competencies
                if competency_activations and synergy_edges:
                    synergy_bonus = self.bridge.competencies.synergy_bias(competency_activations, synergy_edges)
                    
                    # Log synergy events
                    if synergy_bonus > 0.0:
                        if hasattr(self.bridge, 'memory_store') and self.bridge.memory_store is not None:
                            try:
                                if hasattr(self.bridge.memory_store, 'log_thought_event'):
                                    self.bridge.memory_store.log_thought_event({
                                        "type": "competency_synergy",
                                        "edges": [(A, B, round(sim, 4)) for A, B, sim in synergy_edges],
                                        "active": [(name, round(sim, 4)) for name, sim in competency_activations],
                                        "bonus": round(synergy_bonus, 4)
                                    })
                            except Exception:
                                pass
                        
                        # Also log via logger
                        if hasattr(self.bridge, 'logger'):
                            try:
                                self.bridge.logger.write({
                                    "competency_synergy": {
                                        "edges": [(A, B, round(sim, 4)) for A, B, sim in synergy_edges],
                                        "active": [(name, round(sim, 4)) for name, sim in competency_activations],
                                        "bonus": round(synergy_bonus, 4),
                                        "edge_count": len(synergy_edges)
                                    }
                                })
                            except Exception:
                                pass
        except Exception as e:
            # If synergy computation fails, continue without it
            if hasattr(self.bridge, 'logger'):
                try:
                    self.bridge.logger.write({"competency_synergy_error": str(e)})
                except Exception:
                    pass
        
        # Store synergy info for output
        self._synergy_edges = synergy_edges
        self._synergy_bonus = synergy_bonus
        
        # 2. Select the strongest thought
        if not candidates:
            chosen_embedding = None
            dbg = {"note": "No candidates generated"}
        else:
            # Pass competency bias and synergy bias to thought selector
            result = self.bridge.select_thought(candidates, competency_bias=competency_bias, synergy_bias=synergy_bonus)
            
            # Handle both tuple and single return values
            if isinstance(result, tuple):
                chosen_embedding, dbg = result
                if dbg is None:
                    dbg = {"note": "No debug info available"}
            else:
                chosen_embedding = result
                dbg = {"note": "No debug info available"}
        
        # A221 — Update Thought Signature after selection
        signature_vec = None
        signature_preview = None
        if chosen_embedding is not None:
            try:
                signature_vec = self.bridge.thought_signature.update(
                    chosen_embedding,
                    synergy_bonus
                )
                # Attach signature preview to debug info
                if dbg is None:
                    dbg = {}
                if isinstance(signature_vec, list):
                    dbg["thought_signature_preview"] = signature_vec[:8]
                    signature_preview = signature_vec[:12]
                else:
                    # Handle tensor case
                    try:
                        import torch
                        if isinstance(signature_vec, torch.Tensor):
                            dbg["thought_signature_preview"] = signature_vec[:8].tolist()
                            signature_preview = signature_vec[:12].tolist()
                        else:
                            sig_list = list(signature_vec)[:12] if hasattr(signature_vec, '__iter__') else []
                            dbg["thought_signature_preview"] = sig_list[:8]
                            signature_preview = sig_list
                    except:
                        dbg["thought_signature_preview"] = []
                        signature_preview = []
                
                # A221 — Log signature update to memory store
                if signature_preview is not None:
                    try:
                        if hasattr(self.bridge, 'memory_store') and self.bridge.memory_store is not None:
                            if hasattr(self.bridge.memory_store, 'log_thought_event'):
                                self.bridge.memory_store.log_thought_event({
                                    "type": "thought_signature_update",
                                    "signature_preview": signature_preview,
                                    "synergy_bonus": round(synergy_bonus, 4)
                                })
                    except Exception:
                        pass
                
                # Store signature preview for output
                self._signature_preview = signature_preview
            except Exception as e:
                # If signature update fails, continue without it
                if hasattr(self.bridge, 'logger'):
                    try:
                        self.bridge.logger.write({"thought_signature_update_error": str(e)})
                    except Exception:
                        pass
        
        # A222 — Harmonize chosen thought toward ADRAE's identity signature
        if chosen_embedding is not None:
            try:
                if hasattr(self.bridge, 'harmonizer') and self.bridge.harmonizer is not None:
                    harmonized = self.bridge.harmonizer.harmonize(chosen_embedding)
                    if harmonized is not None:
                        chosen_embedding = harmonized
            except Exception as e:
                # If harmonization fails, continue with original chosen_embedding
                if hasattr(self.bridge, 'logger'):
                    try:
                        self.bridge.logger.write({"chosen_thought_harmonization_error": str(e)})
                    except Exception:
                        pass
        
        # A223 — Apply personality flow field to chosen thought
        if chosen_embedding is not None:
            try:
                if hasattr(self.bridge, 'flow') and self.bridge.flow is not None:
                    flow_applied = self.bridge.flow.apply_flow(chosen_embedding)
                    if flow_applied is not None:
                        chosen_embedding = flow_applied
            except Exception as e:
                # If flow application fails, continue with original chosen_embedding
                if hasattr(self.bridge, 'logger'):
                    try:
                        self.bridge.logger.write({"flow_application_error": str(e)})
                    except Exception:
                        pass
        
        # A223 — Update flow field with chosen thought
        if chosen_embedding is not None:
            try:
                if hasattr(self.bridge, 'flow') and self.bridge.flow is not None:
                    self.bridge.flow.update_flow(chosen_embedding)
            except Exception as e:
                # If flow update fails, continue without it
                if hasattr(self.bridge, 'logger'):
                    try:
                        self.bridge.logger.write({"flow_update_error": str(e)})
                    except Exception:
                        pass

        # -----------------------------------------
        # 🔥 CRITICAL FIX: Inject chosen thought
        # Without this, PRIME never forms state,
        # never updates fusion, never updates attention,
        # and cognition stays at zero forever.
        # -----------------------------------------
        if chosen_embedding is not None:
            try:
                # Update neural internal state (updates drift and timescales)
                self.bridge.state.update(chosen_embedding)

                # A183 — Capture baseline identity once it's formed
                if self.bridge.baseline_identity is None:
                    identity_vec = self.bridge.state.timescales.identity_vector
                    if identity_vec is not None:
                        try:
                            from ..neural.torch_utils import safe_tensor
                            import torch
                            identity_tensor = safe_tensor(identity_vec)
                            if identity_tensor is not None:
                                if isinstance(identity_tensor, torch.Tensor):
                                    self.bridge.baseline_identity = identity_tensor.detach().clone()
                                else:
                                    # Fallback for lists
                                    self.bridge.baseline_identity = list(identity_vec) if hasattr(identity_vec, '__iter__') else identity_vec
                        except Exception as e:
                            if hasattr(self.bridge, 'logger'):
                                self.bridge.logger.write({"baseline_capture_error": str(e)})

                # Recompute attention with updated timescales
                if hasattr(self.bridge, "attention"):
                    self.bridge.attention.compute_attention_vector(self.bridge.state.timescales)

                # Recompute fusion with updated attention and timescales
                if hasattr(self.bridge, "fusion"):
                    self.bridge.fusion.fuse(
                        self.bridge.attention.last_focus_vector,
                        self.bridge.state.timescales
                    )
                    
                    # A-SOV-05: Inject ADRAE identity into workspace
                    try:
                        self.bridge.adrae_workspace.inject_identity()
                    except Exception as e:
                        if hasattr(self.bridge, 'logger'):
                            self.bridge.logger.write({"adrae_workspace_injection_error": str(e)})
                    
                    # A202 — Update Global Workspace Continuity
                    try:
                        drift_state = self.bridge.state.drift.get_status()
                        drift_level = drift_state.get("avg_drift", 0.0) if drift_state else 0.0
                        attention_vec = self.bridge.attention.last_focus_vector
                        continuity_vec = self.bridge.continuity.update(
                            chosen_embedding,
                            attention_vec,
                            drift_level
                        )
                        # Store continuity in output (will be added to last_output later)
                        self._continuity_vec = continuity_vec
                    except Exception as e:
                        if hasattr(self.bridge, 'logger'):
                            self.bridge.logger.write({"continuity_update_error": str(e)})
                        self._continuity_vec = None
                    
                    # === A165: Personality Drift Regulation ===
                    identity_vec = self.bridge.state.timescales.identity_vector
                    
                    # A183 — Identity drift prevention
                    if self.bridge.baseline_identity is not None and identity_vec is not None:
                        try:
                            corrected_identity, drift_value = self.bridge.identity_drift.correct(
                                identity_vec,
                                self.bridge.baseline_identity
                            )
                            
                            # Apply correction if drift was detected
                            if corrected_identity is not None and corrected_identity is not identity_vec:
                                # Update identity vector with corrected version
                                from ..neural.torch_utils import safe_tensor
                                import torch
                                corrected_tensor = safe_tensor(corrected_identity)
                                if corrected_tensor is not None:
                                    if isinstance(corrected_tensor, torch.Tensor):
                                        self.bridge.state.timescales.identity_vector = corrected_tensor
                                    else:
                                        self.bridge.state.timescales.identity_vector = corrected_identity
                            
                            # Log drift for diagnostics
                            if hasattr(self.bridge, 'memory_store') and self.bridge.memory_store is not None:
                                try:
                                    if hasattr(self.bridge.memory_store, 'log_identity_drift'):
                                        self.bridge.memory_store.log_identity_drift({
                                            "drift_value": drift_value,
                                            "within_limits": drift_value < self.bridge.identity_drift.max_drift
                                        })
                                    elif hasattr(self.bridge, 'logger'):
                                        # Fallback to logger if memory_store doesn't have log_identity_drift
                                        self.bridge.logger.write({
                                            "identity_drift": {
                                                "drift_value": drift_value,
                                                "within_limits": drift_value < self.bridge.identity_drift.max_drift
                                            }
                                        })
                                except Exception:
                                    # If logging fails, continue without it
                                    pass
                        except Exception as e:
                            if hasattr(self.bridge, 'logger'):
                                self.bridge.logger.write({"identity_drift_correction_error": str(e)})
                    
                    # A184 — Log identity coherence
                    if self.bridge.baseline_identity is not None and identity_vec is not None:
                        try:
                            sim = self.bridge.identity_gate.cosine_sim(
                                identity_vec,
                                self.bridge.baseline_identity
                            )
                            
                            # Log to memory_store if available
                            if hasattr(self.bridge, 'memory_store') and self.bridge.memory_store is not None:
                                try:
                                    if hasattr(self.bridge.memory_store, 'log_identity_coherence'):
                                        self.bridge.memory_store.log_identity_coherence({
                                            "similarity_to_baseline": sim
                                        })
                                    elif hasattr(self.bridge, 'logger'):
                                        # Fallback to logger
                                        self.bridge.logger.write({
                                            "identity_coherence": {
                                                "similarity_to_baseline": sim
                                            }
                                        })
                                except Exception:
                                    # If logging fails, continue without it
                                    pass
                        except Exception as e:
                            if hasattr(self.bridge, 'logger'):
                                self.bridge.logger.write({"identity_coherence_log_error": str(e)})
                    
                    personality_vec = self.bridge.fusion.last_fusion_vector
                    
                    if identity_vec is not None and personality_vec is not None:
                        try:
                            regulated, drift_level, drift_value = \
                                self.bridge.personality.update(identity_vec, personality_vec)
                            
                            # Replace personality vector if changed
                            self.bridge.fusion.last_fusion_vector = regulated
                            
                            # === A166 — Personality Gradient Learning ===
                            shaped = regulated  # Default to regulated if gradient fails
                            try:
                                # Record personality vector for gradient learning
                                self.bridge.personality_gradient.record(regulated)
                                
                                # Apply gradient shaping to personality
                                shaped = self.bridge.personality_gradient.apply_gradient(regulated)
                            except Exception as e:
                                # If gradient application fails, keep regulated vector
                                if hasattr(self.bridge, 'logger'):
                                    self.bridge.logger.write({"personality_gradient_error": str(e)})
                            
                            # === A167 — Update Personality Signature ===
                            signature = None
                            try:
                                signature = self.bridge.personality_signature.update(shaped)
                                
                                # Apply signature influence to the fusion state
                                shaped_with_signature = self.bridge.personality_signature.apply(shaped)
                                self.bridge.fusion.last_fusion_vector = shaped_with_signature
                                
                                # Store signature status for output
                                self._personality_signature_info = {
                                    "active": signature is not None
                                }
                            except Exception as e:
                                # If signature application fails, keep shaped vector
                                if hasattr(self.bridge, 'logger'):
                                    self.bridge.logger.write({"personality_signature_error": str(e)})
                                self.bridge.fusion.last_fusion_vector = shaped
                                self._personality_signature_info = {"active": False}
                            
                            # === A168 — Lifelong Personality Continuity Engine ===
                            try:
                                stable_signature = self.bridge.personality_continuity.update(signature)
                                # If continuity stabilized the signature, update fusion vector
                                if stable_signature is not None and stable_signature is not signature:
                                    # Re-apply signature shaping with stabilized signature
                                    try:
                                        shaped_with_stable = self.bridge.personality_signature.apply(stable_signature)
                                        self.bridge.fusion.last_fusion_vector = shaped_with_stable
                                    except Exception:
                                        pass  # If re-application fails, keep current vector
                            except Exception as e:
                                # If continuity fails, continue without it
                                if hasattr(self.bridge, 'logger'):
                                    self.bridge.logger.write({"personality_continuity_error": str(e)})
                            
                            # Store drift info for output (will be added to last_output later)
                            self._personality_drift_info = {
                                "level": drift_level,
                                "value": drift_value
                            }
                        except Exception as e:
                            # If regulation fails, continue without it
                            if hasattr(self.bridge, 'logger'):
                                self.bridge.logger.write({"personality_regulation_error": str(e)})
                            self._personality_drift_info = None
                    else:
                        self._personality_drift_info = None
            except Exception as e:
                print("🔥 ERROR: Failed to update cognition state:", e)
                self._personality_drift_info = None

        # ---------------------------------------------
        # A179 — Supervisory evaluation + A180 — Conflict assessment + A181 — Meta-intent
        # ---------------------------------------------
        supervision = None
        conflict_resolution = 0.5  # Default neutral resolution
        intent_vector = None
        
        if chosen_embedding is not None:
            try:
                supervision = self.bridge.supervisor.supervise(chosen_embedding, self.bridge.state)
            except Exception as e:
                if hasattr(self.bridge, 'logger'):
                    self.bridge.logger.write({"supervision_error": str(e)})
                supervision = None
        
        # A180 — Detect and resolve conflicts
        try:
            conflict = self.bridge.conflict_resolver.detect_conflict(self.bridge.state, self.bridge)
            conflict_resolution = self.bridge.conflict_resolver.resolve(conflict)
            
            # Store evolution pressure back into state
            if not hasattr(self.bridge.state, 'evolution_pressure'):
                self.bridge.state.evolution_pressure = conflict_resolution
            else:
                self.bridge.state.evolution_pressure = conflict_resolution
        except Exception as e:
            if hasattr(self.bridge, 'logger'):
                self.bridge.logger.write({"conflict_resolution_error": str(e)})
            conflict_resolution = 0.5  # Fallback to neutral
        
        # A181 — Compute intent weights and combine intent vectors
        try:
            # Operator context (tasks)
            operator_context = []
            if hasattr(self.bridge, 'tasks') and self.bridge.tasks is not None:
                if hasattr(self.bridge.tasks, 'list_tasks'):
                    operator_context = self.bridge.tasks.list_tasks()
                elif hasattr(self.bridge.tasks, 'queue'):
                    # Fallback: extract tasks from queue directly
                    operator_context = [task for _, _, task in self.bridge.tasks.queue]
            
            # Compute intent weighting
            intent_weights = self.bridge.meta_intent.compute_intent_weights(
                self.bridge.state,
                operator_context
            )
            
            # Get vectors for intent combination
            operator_vec = chosen_embedding  # Use chosen thought as operator intent proxy
            system_vec = None
            self_vec = None
            
            # System intent: identity vector
            try:
                if hasattr(self.bridge.state, 'timescales') and self.bridge.state.timescales is not None:
                    system_vec = getattr(self.bridge.state.timescales, 'identity_vector', None)
            except Exception:
                pass
            
            # Self intent: evolution vector or identity vector as fallback
            try:
                # Try to get evolution vector from trajectory
                if hasattr(self, 'last_output') and self.last_output:
                    trajectory = self.last_output.get("evolution_trajectory", {})
                    self_vec = trajectory.get("vector")
                
                # Fallback to identity vector if evolution vector not available
                if self_vec is None and system_vec is not None:
                    self_vec = system_vec
            except Exception:
                pass
            
            # Combine intent vectors into a single intent bias signal
            intent_vector = self.bridge.meta_intent.combine_intents(
                operator_vec=operator_vec,
                system_vec=system_vec,
                self_vec=self_vec
            )
        except Exception as e:
            if hasattr(self.bridge, 'logger'):
                self.bridge.logger.write({"meta_intent_error": str(e)})
            intent_vector = None
        
        # A182 — reinforce workspace pathways based on intent vs chosen embedding (feedback learning)
        if chosen_embedding is not None and intent_vector is not None:
            try:
                self.bridge.workspace_reinforcement.reinforce(
                    intent_vector=intent_vector,
                    workspace_vector=chosen_embedding
                )
            except Exception as e:
                if hasattr(self.bridge, 'logger'):
                    self.bridge.logger.write({"reinforcement_feedback_error": str(e)})
        
        # === A207: Shape cognitive path based on goal alignment ===
        path_state = None
        try:
            harmonized_goal = getattr(self.bridge, 'last_harmonized_goal', None)
            fusion_vec = self.bridge.fusion.last_fusion_vector
            action_weights = self.bridge.action_engine.action_weights.copy() if hasattr(self.bridge.action_engine, 'action_weights') else {}
            
            if harmonized_goal is not None and fusion_vec is not None:
                path_state = self.bridge.path_shaper.shape_path(
                    harmonized_goal,
                    fusion_vec,
                    action_weights
                )
                
                # Apply updated weights to action engine
                if path_state and path_state.get("updated_weights"):
                    # Temporarily update action weights for this cycle
                    original_weights = self.bridge.action_engine.action_weights.copy()
                    self.bridge.action_engine.action_weights.update(path_state["updated_weights"])
                    self._original_action_weights = original_weights  # Store for restoration
        except Exception as e:
            # If path shaping fails, continue without it
            if hasattr(self.bridge, 'logger'):
                self.bridge.logger.write({"path_shaping_error": str(e)})
            path_state = None
        
        # Store path_state for output
        self._path_state = path_state
        
        # Choose action with bias from supervisor, conflict resolution, and intent vector
        if supervision and "action_bias" in supervision:
            action = self.bridge.action_engine.choose_biased(
                supervision["action_bias"], 
                conflict_resolution,
                intent_vector,
                self.bridge
            )
        else:
            action = self.bridge.choose_cognitive_action()
        
        # Restore original action weights after selection (A207)
        if hasattr(self, '_original_action_weights'):
            self.bridge.action_engine.action_weights = self._original_action_weights
            delattr(self, '_original_action_weights')
        
        # === A208: Generate adaptive subgoals ===
        subgoal_state = None
        competition_state = None  # Initialize before try block
        routing_state = None  # Initialize before try block
        planning_state = None  # Initialize before try block
        try:
            harmonized_goal = getattr(self.bridge, 'last_harmonized_goal', None)
            fusion_vec = self.bridge.fusion.last_fusion_vector
            momentum_vec = None
            
            # Get momentum from path shaper
            if hasattr(self.bridge.path_shaper, 'momentum'):
                momentum_vec = self.bridge.path_shaper.momentum
            
            if harmonized_goal is not None and fusion_vec is not None:
                # Generate subgoals
                subgoal_state = self.bridge.subgoal_generator.generate(
                    harmonized_goal,
                    fusion_vec,
                    momentum_vec
                )
                
                # === A209: Run subgoal competition ===
                competition_state = None
                try:
                    active_subgoals = self.bridge.subgoal_generator.active_subgoals
                    drift_value = None
                    if hasattr(self.bridge.state, 'drift'):
                        drift_state = self.bridge.state.drift.get_status()
                        drift_value = drift_state.get("latest_drift", 0.0) if drift_state else 0.0
                    
                    if active_subgoals and len(active_subgoals) > 0:
                        competition_state = self.bridge.subgoal_competition.compete(
                            active_subgoals,
                            harmonized_goal,
                            fusion_vec,
                            drift_value
                        )
                        
                        # Apply competition winner's influence to fusion vector
                        if competition_state and competition_state.get("competition_vector") is not None:
                            modified_fusion = self.bridge.subgoal_competition.apply_competition(
                                fusion_vec,
                                competition_state["competition_vector"]
                            )
                            # Update fusion vector with competition winner
                            if modified_fusion is not None:
                                self.bridge.fusion.last_fusion_vector = modified_fusion
                            
                            # === A210: Route subgoals into execution pathways ===
                            try:
                                active_subgoals = self.bridge.subgoal_generator.active_subgoals
                                attention_vec = self.bridge.attention.last_focus_vector
                                
                                if active_subgoals and len(active_subgoals) > 0:
                                    routing_state = self.bridge.subgoal_router.route(
                                        active_subgoals,
                                        harmonized_goal,
                                        self.bridge.fusion.last_fusion_vector,  # Use updated fusion
                                        attention_vec,
                                        drift_value
                                    )
                                    
                                    # Apply routing modifications to fusion and attention
                                    if routing_state:
                                        mod_fusion = routing_state.get("modified_fusion")
                                        mod_attention = routing_state.get("modified_attention")
                                        
                                        if mod_fusion is not None:
                                            self.bridge.fusion.last_fusion_vector = mod_fusion
                                        
                                        if mod_attention is not None:
                                            self.bridge.attention.last_focus_vector = mod_attention
                                        
                                        # === A211: Build multi-step execution chain ===
                                        try:
                                            # Get current route and active subgoals
                                            current_route = routing_state.get("current_route")
                                            active_subgoals = self.bridge.subgoal_generator.active_subgoals
                                            
                                            if current_route and active_subgoals and len(active_subgoals) >= 2:
                                                planning_state = self.bridge.planning_engine.plan(
                                                    active_subgoals,
                                                    current_route,
                                                    self.bridge.fusion.last_fusion_vector,
                                                    self.bridge.attention.last_focus_vector
                                                )
                                                
                                                # Apply planning modifications to fusion and attention
                                                if planning_state and planning_state.get("plan_valid"):
                                                    plan_fusion = planning_state.get("modified_fusion")
                                                    plan_attention = planning_state.get("modified_attention")
                                                    
                                                    if plan_fusion is not None:
                                                        self.bridge.fusion.last_fusion_vector = plan_fusion
                                                    
                                                    if plan_attention is not None:
                                                        self.bridge.attention.last_focus_vector = plan_attention
                                        except Exception as e:
                                            # If planning fails, continue without it
                                            if hasattr(self.bridge, 'logger'):
                                                self.bridge.logger.write({"planning_error": str(e)})
                                            planning_state = None
                                        
                                        # Store planning state for output
                                        self._planning_state = planning_state
                                        
                                        # A212 — Update active chain from planning state
                                        if planning_state and planning_state.get("plan_valid"):
                                            execution_order = planning_state.get("execution_order", [])
                                            if execution_order:
                                                # Build chain from execution order
                                                chain = []
                                                for step_info in execution_order:
                                                    step_id = step_info.get("id")
                                                    # Find matching subgoal
                                                    matching_sg = next((sg for sg in active_subgoals if sg.get("id") == step_id), None)
                                                    if matching_sg:
                                                        chain.append(matching_sg)
                                                
                                                if chain:
                                                    self.active_chain = chain
                                                    self.chain_step_index = 0
                                                    self.chain_failures = 0
                            except Exception as e:
                                # If routing fails, continue without it
                                if hasattr(self.bridge, 'logger'):
                                    self.bridge.logger.write({"subgoal_routing_error": str(e)})
                                routing_state = None
                                self._planning_state = None
                        else:
                            # Fallback to A208 subgoal influence if no competition winner
                            if subgoal_state and subgoal_state.get("subgoal_influence") is not None:
                                modified_fusion = self.bridge.subgoal_generator.apply_influence(
                                    fusion_vec,
                                    subgoal_state["subgoal_influence"]
                                )
                                if modified_fusion is not None:
                                    self.bridge.fusion.last_fusion_vector = modified_fusion
                except Exception as e:
                    # If competition fails, fallback to A208 subgoal influence
                    if hasattr(self.bridge, 'logger'):
                        self.bridge.logger.write({"subgoal_competition_error": str(e)})
                    competition_state = None
                    
                    # Apply subgoal influence as fallback
                    if subgoal_state and subgoal_state.get("subgoal_influence") is not None:
                        try:
                            modified_fusion = self.bridge.subgoal_generator.apply_influence(
                                fusion_vec,
                                subgoal_state["subgoal_influence"]
                            )
                            if modified_fusion is not None:
                                self.bridge.fusion.last_fusion_vector = modified_fusion
                        except Exception:
                            pass
        except Exception as e:
            # If subgoal generation fails, continue without it
            if hasattr(self.bridge, 'logger'):
                self.bridge.logger.write({"subgoal_generation_error": str(e)})
            subgoal_state = None
            competition_state = None
            routing_state = None
            planning_state = None
        
        # Store subgoal, competition, routing, and planning state for output
        self._subgoal_state = subgoal_state
        self._competition_state = competition_state
        self._routing_state = routing_state
        self._planning_state = planning_state
        
        # ---------------------------------------------
        # A163 — Evolution-biased cognitive action selection (still applies)
        # ---------------------------------------------
        if trend == "upward":
            # More identity growth + reflection when healthy
            if action == "retrieve_memory":
                action = "generate_reflection"
        elif trend == "unstable":
            # More stabilization behavior
            if action == "generate_reflection":
                action = "retrieve_memory"
        
        # ---------------------------------------------
        # A164 — Get cognitive mode from scheduler (early check)
        # ---------------------------------------------
        # Get drift for early scheduler check
        drift_state_temp = self.bridge.state.drift.get_status()
        drift_value_early = drift_state_temp.get("latest_drift", 0.0) if drift_state_temp else 0.0
        
        # Use placeholder coherence for early check (will be updated with actual value later)
        coherence_value_early = 1.0
        
        # Get cognitive mode from scheduler (early check for action override)
        mode_early = self.bridge.growth.update(
            drift_value_early,
            coherence_value_early,
            trend
        )
        
        # Mode overrides action
        if mode_early == "stabilize":
            action = "update_identity"
        elif mode_early == "reflect":
            action = "generate_reflection"
        elif mode_early == "evolve":
            # Allow evolutionary bias to remain
            pass
        elif mode_early == "idle":
            # Keep current action
            pass
        
        # 3. Execute a cognitive action
        action_output = self.bridge.perform_action(action)
        
        # 3b. A212 — Monitor multi-step execution chain
        chain_update = None
        try:
            chain_update = self._monitor_and_reroute(
                action_output if isinstance(action_output, dict) else {}
            )
        except Exception as e:
            # If monitoring fails, continue without it
            if hasattr(self.bridge, 'logger'):
                self.bridge.logger.write({"chain_monitoring_error": str(e)})
            chain_update = {"status": "monitoring_error"}
        
        # Store chain update for output
        self._chain_update = chain_update
        
        # 4. Memory metabolism
        recalled = self.bridge.memory_cycle()

        # ---------------------------------------------
        # A163 — Growth-aligned memory reinforcement
        # ---------------------------------------------
        if evo_vec is not None and hasattr(self.bridge.state, "memory_manager") and self.bridge.state.memory_manager:
            try:
                from ..neural.torch_utils import safe_tensor, is_tensor
                evo_tensor = safe_tensor(evo_vec)
                if is_tensor(evo_tensor):
                    # Reinforce memories that align with evolutionary direction
                    # This is a simplified approach - in a full implementation,
                    # we'd have a reinforce_direction method
                    # For now, we'll use semantic memory's find_similar to reinforce
                    if hasattr(self.bridge.state.memory_manager, 'semantic'):
                        # Find memories similar to evolution vector and reinforce them
                        similar = self.bridge.state.memory_manager.semantic.find_similar(evo_tensor, top_k=3)
                        # The reinforcement happens implicitly through access
            except Exception as e:
                if hasattr(self.bridge, 'logger'):
                    self.bridge.logger.write({"evo_memory_reinforce_error": str(e)})

        # 5. Update attention & fusion (internal to bridge)
        # Fusion gets recomputed automatically on next perception/reflection

        # 6. Drift, coherence, and uncertainty (A216)
        drift_state = self.bridge.state.drift.get_status()
        fusion_state = self.bridge.fusion.status()
        attention_state = self.bridge.attention.status()
        
        # A216 — Compute uncertainty signal
        uncertainty = self.bridge.state.compute_uncertainty(
            self.bridge.fusion.last_fusion_vector,
            self.bridge.attention.last_focus_vector,
            drift_state
        )
        
        # Store uncertainty for action engine and output
        self.bridge.state.last_uncertainty = uncertainty
        self._uncertainty_value = uncertainty  # Store for output
        
        # A217 — If uncertainty is high, generate a new skill vector
        if uncertainty > 0.65:
            try:
                # Extract skill vector from current fusion state
                skill_vec = self.bridge.fusion.last_fusion_vector
                if skill_vec is not None:
                    # Generate unique skill name
                    import time
                    skill_name = f"skill_auto_{int(time.time())}"
                    
                    # Create metadata about the skill
                    skill_metadata = {
                        "uncertainty": uncertainty,
                        "drift": drift_state.get("latest_drift", 0.0) if drift_state else 0.0,
                        "fusion_coherence": fusion_state.get("coherence", 1.0) if isinstance(fusion_state, dict) else 1.0,
                        "attention_strength": attention_state.get("strength", 0.0) if isinstance(attention_state, dict) else 0.0,
                        "created_at": time.time()
                    }
                    
                    # Add skill to skill bank
                    skill_entry = self.bridge.skills.add_skill(skill_name, skill_vec, skill_metadata)
                    
                    if skill_entry:
                        # Log skill creation
                        if hasattr(self.bridge, 'memory_store') and self.bridge.memory_store is not None:
                            try:
                                if hasattr(self.bridge.memory_store, 'log_thought_event'):
                                    self.bridge.memory_store.log_thought_event({
                                        "type": "new_skill_created",
                                        "skill": skill_name,
                                        "uncertainty": uncertainty,
                                        "skill_count": self.bridge.skills.get_skill_count()
                                    })
                            except Exception:
                                pass
                        
                        # Also log via logger if available
                        if hasattr(self.bridge, 'logger'):
                            try:
                                self.bridge.logger.write({
                                    "new_skill_created": {
                                        "skill": skill_name,
                                        "uncertainty": round(uncertainty, 4),
                                        "skill_count": self.bridge.skills.get_skill_count()
                                    }
                                })
                            except Exception:
                                pass
            except Exception as e:
                # If skill creation fails, continue without it
                if hasattr(self.bridge, 'logger'):
                    try:
                        self.bridge.logger.write({"skill_creation_error": str(e)})
                    except Exception:
                        pass
        
        # Log uncertainty
        if hasattr(self.bridge, 'memory_store') and self.bridge.memory_store is not None:
            try:
                if hasattr(self.bridge.memory_store, 'log_thought_event'):
                    self.bridge.memory_store.log_thought_event({
                        "type": "uncertainty",
                        "value": uncertainty
                    })
                elif hasattr(self.bridge, 'logger'):
                    self.bridge.logger.write({
                        "uncertainty": {
                            "value": round(uncertainty, 4),
                            "level": "low" if uncertainty < 0.25 else "moderate" if uncertainty < 0.5 else "high" if uncertainty < 0.75 else "extreme"
                        }
                    })
            except Exception:
                # If logging fails, continue without it
                pass
        
        # A213 — Store completed or rerouted chains (after drift/coherence computed)
        if chain_update and chain_update.get("status") in ("chain_complete", "chain_rerouted"):
            try:
                result = {
                    "status": chain_update.get("status"),
                    "reroutes": self.chain_failures,
                    "avg_drift": drift_state.get("avg_drift", 0.0) if isinstance(drift_state, dict) else 0.0,
                    "coherence": fusion_state.get("coherence", 1.0) if isinstance(fusion_state, dict) else 1.0,
                }
                
                # Get chain to store
                chain_to_store = None
                if chain_update.get("status") == "chain_complete":
                    chain_to_store = chain_update.get("completed_chain")
                elif chain_update.get("status") == "chain_rerouted":
                    # Use current active chain if available
                    chain_to_store = self.active_chain
                
                if chain_to_store:
                    stored_entry = self.bridge.chain_memory.store_chain(chain_to_store, result)
                    
                    # A214 — Convert strong chains into reusable skills
                    if stored_entry and stored_entry.get("score", 0.0) > 0.75:  # Threshold for expertise formation
                        try:
                            skill_entry = self.bridge.skill_encoder.encode_chain(
                                stored_entry.get("chain", []),
                                stored_entry
                            )
                            if skill_entry:
                                # Update generator with skill embeddings
                                if hasattr(self.bridge, 'generator'):
                                    skill_embeddings = self.bridge.skill_encoder.get_skill_embeddings()
                                    self.bridge.generator.skill_embeddings = skill_embeddings
                                
                                # A215 — Extract abstract skill pattern
                                try:
                                    pattern = self.bridge.skill_generalizer.extract_pattern(
                                        skill_entry.get("skill_vector")
                                    )
                                    if pattern is not None:
                                        # Store generalizable skill
                                        self.bridge.skill_generalizer.register_generalized_skill(
                                            pattern,
                                            stored_entry
                                        )
                                        
                                        # Update generator with generalized patterns
                                        if hasattr(self.bridge, 'generator'):
                                            generalized_patterns = [
                                                g.get("pattern")
                                                for g in self.bridge.skill_generalizer.get_generalized_skills()
                                                if g.get("pattern") is not None
                                            ]
                                            self.bridge.generator.generalized_skill_patterns = generalized_patterns
                                        
                                        # Log generalization
                                        if hasattr(self.bridge, 'logger'):
                                            self.bridge.logger.write({
                                                "skill_generalized": {
                                                    "score": round(stored_entry.get("score", 0.0), 4),
                                                    "total_generalized": len(self.bridge.skill_generalizer.generalized_skills)
                                                }
                                            })
                                except Exception as gen_e:
                                    # If generalization fails, continue without it
                                    if hasattr(self.bridge, 'logger'):
                                        self.bridge.logger.write({"skill_generalization_error": str(gen_e)})
                                
                                # Log skill encoding
                                if hasattr(self.bridge, 'logger'):
                                    self.bridge.logger.write({
                                        "skill_encoded": {
                                            "score": round(stored_entry.get("score", 0.0), 4),
                                            "chain_length": len(stored_entry.get("chain", [])),
                                            "total_skills": len(self.bridge.skill_encoder.skills)
                                        }
                                    })
                        except Exception as e:
                            # If skill encoding fails, continue without it
                            if hasattr(self.bridge, 'logger'):
                                self.bridge.logger.write({"skill_encoding_error": str(e)})
                    
                    # Optional: chain optimization pass (every 10 chains)
                    if self.bridge.chain_memory.get_chain_count() % 10 == 0:
                        opt_result = self.bridge.chain_memory.optimize()
                        if opt_result and opt_result.get("pruned", 0) > 0:
                            if hasattr(self.bridge, 'logger'):
                                self.bridge.logger.write({
                                    "chain_optimization": {
                                        "pruned": opt_result["pruned"],
                                        "remaining": opt_result["remaining"],
                                        "threshold": round(opt_result["threshold"], 4)
                                    }
                                })
            except Exception as e:
                # If chain storage fails, continue without it
                if hasattr(self.bridge, 'logger'):
                    self.bridge.logger.write({"chain_storage_error": str(e)})
        
        # ---------------------------------------------------
        # A185 — Sleep/Wake Identity Consolidation Trigger
        # ---------------------------------------------------
        self.bridge.cycle_step += 1

        if self.bridge.cycle_step >= self.bridge.sleep_cycle_interval:
            # Begin "sleep" cycle - consolidate identity
            baseline = self.bridge.baseline_identity
            current = self.bridge.state.timescales.identity_vector

            if baseline is not None and current is not None:
                try:
                    consolidated = self.bridge.identity_consolidator.consolidate(
                        current,
                        baseline,
                        self.bridge.identity_drift
                    )
                    
                    if consolidated is not None:
                        # Apply consolidated identity
                        from ..neural.torch_utils import safe_tensor
                        import torch
                        consolidated_tensor = safe_tensor(consolidated)
                        if consolidated_tensor is not None:
                            if isinstance(consolidated_tensor, torch.Tensor):
                                self.bridge.state.timescales.identity_vector = consolidated_tensor
                            else:
                                self.bridge.state.timescales.identity_vector = consolidated
                        
                        # Log consolidation event
                        try:
                            baseline_similarity = self.bridge.identity_gate.cosine_sim(
                                consolidated,
                                baseline
                            )
                            
                            if hasattr(self.bridge, 'memory_store') and self.bridge.memory_store is not None:
                                try:
                                    if hasattr(self.bridge.memory_store, 'log_identity_consolidation'):
                                        self.bridge.memory_store.log_identity_consolidation({
                                            "event": "identity_consolidated",
                                            "baseline_similarity": baseline_similarity
                                        })
                                    elif hasattr(self.bridge, 'logger'):
                                        # Fallback to logger
                                        self.bridge.logger.write({
                                            "identity_consolidation": {
                                                "event": "identity_consolidated",
                                                "baseline_similarity": baseline_similarity
                                            }
                                        })
                                except Exception:
                                    pass
                        except Exception as e:
                            if hasattr(self.bridge, 'logger'):
                                self.bridge.logger.write({"consolidation_log_error": str(e)})
                except Exception as e:
                    if hasattr(self.bridge, 'logger'):
                        self.bridge.logger.write({"identity_consolidation_error": str(e)})

            # -------------------------------------------
            # A186 — Dreamspace Activation
            # -------------------------------------------
            try:
                mm = self.bridge.state.memory_manager if hasattr(self.bridge.state, "memory_manager") else None
                id_vec = self.bridge.state.timescales.identity_vector

                if id_vec is not None:
                    dream_events = []
                    
                    # Generate 3 dream events
                    for _ in range(3):
                        d = self.bridge.dreamspace.generate_dream(mm, id_vec)
                        if d is not None:
                            dream_events.append(d)
                            
                            # Use dream as subconscious reinforcement
                            # Blend dream with identity (subtle influence)
                            from ..neural.torch_utils import safe_tensor, TORCH_AVAILABLE
                            import torch
                            dream_tensor = safe_tensor(d)
                            id_tensor = safe_tensor(id_vec)
                            
                            if dream_tensor is not None and id_tensor is not None:
                                if TORCH_AVAILABLE and isinstance(dream_tensor, torch.Tensor) and isinstance(id_tensor, torch.Tensor):
                                    if dream_tensor.shape == id_tensor.shape:
                                        # Subtle blend: 95% identity, 5% dream
                                        id_vec = (id_tensor * 0.95 + dream_tensor * 0.05)
                                        norm = torch.norm(id_vec)
                                        if norm > 0:
                                            id_vec = id_vec / norm
                                else:
                                    # Fallback for lists
                                    import math
                                    id_list = list(id_tensor) if hasattr(id_tensor, '__iter__') else [id_tensor]
                                    dream_list = list(dream_tensor) if hasattr(dream_tensor, '__iter__') else [dream_tensor]
                                    
                                    if len(id_list) == len(dream_list):
                                        id_vec = [i * 0.95 + d * 0.05 for i, d in zip(id_list, dream_list)]
                                        norm = math.sqrt(sum(x * x for x in id_vec))
                                        if norm > 0:
                                            id_vec = [v / norm for v in id_vec]

                    # Log dreams
                    if dream_events:
                        try:
                            # Convert dream events to serializable format
                            dream_data = []
                            for d in dream_events:
                                d_tensor = safe_tensor(d)
                                if d_tensor is not None:
                                    if TORCH_AVAILABLE and isinstance(d_tensor, torch.Tensor):
                                        dream_data.append(d_tensor.tolist())
                                    else:
                                        dream_data.append(list(d) if hasattr(d, '__iter__') else [d])
                            
                            if hasattr(self.bridge, 'memory_store') and self.bridge.memory_store is not None:
                                try:
                                    if hasattr(self.bridge.memory_store, 'log_dream_events'):
                                        self.bridge.memory_store.log_dream_events(dream_data)
                                    elif hasattr(self.bridge, 'logger'):
                                        # Fallback to logger
                                        self.bridge.logger.write({
                                            "dream_events": {
                                                "count": len(dream_data),
                                                "events": dream_data[:3]  # Log first 3 for brevity
                                            }
                                        })
                                except Exception:
                                    pass
                        except Exception as e:
                            if hasattr(self.bridge, 'logger'):
                                self.bridge.logger.write({"dream_logging_error": str(e)})

                    # Apply post-dream identity (subtle influence from dreams)
                    if id_vec is not None and id_vec is not self.bridge.state.timescales.identity_vector:
                        id_tensor = safe_tensor(id_vec)
                        if id_tensor is not None:
                            if TORCH_AVAILABLE and isinstance(id_tensor, torch.Tensor):
                                self.bridge.state.timescales.identity_vector = id_tensor
                            else:
                                self.bridge.state.timescales.identity_vector = id_vec
            except Exception as e:
                if hasattr(self.bridge, 'logger'):
                    self.bridge.logger.write({"dreamspace_error": str(e)})

            # Reset cycle counter
            self.bridge.cycle_step = 0
        
        # === A170: Autobiographical Memory Logging ===
        try:
            identity_vec = self.bridge.state.timescales.identity_vector
            long_horizon_vec = getattr(self.bridge.state.timescales, "long_horizon_identity", None)
            
            self.bridge.autobio.record_event(
                event_type=action,
                identity_vec=identity_vec,
                long_horizon_vec=long_horizon_vec,
                reflection=action_output if action == "generate_reflection" else None,
                drift=drift_state
            )
        except Exception as e:
            # If autobiographical logging fails, continue without it
            if hasattr(self.bridge, 'logger'):
                self.bridge.logger.write({"autobiographical_memory_error": str(e)})
        
        # === A171: Update Emergent Self-Model ===
        try:
            autobio_recent = self.bridge.autobio.get_recent(5)
            identity_vec_self = self.bridge.state.timescales.identity_vector
            long_horizon_vec_self = getattr(self.bridge.state.timescales, "long_horizon_identity", None)
            reflection_vec_self = action_output if action == "generate_reflection" else None
            
            self_state = self.bridge.self_model.compute_self_state(
                identity_vec_self,
                long_horizon_vec_self,
                autobio_recent,
                drift_state,
                reflection_vec_self
            )
            
            # Store self-state in output for visibility
            self._self_state = self_state
        except Exception as e:
            # If self-model computation fails, continue without it
            if hasattr(self.bridge, 'logger'):
                self.bridge.logger.write({"self_model_error": str(e)})
            self._self_state = None
        
        # A164: Get actual coherence value for logging
        coherence_value_final = fusion_state.get("coherence", 1.0) if isinstance(fusion_state, dict) else 1.0
        drift_value_final = drift_state.get("latest_drift", 0.0) if drift_state else 0.0
        
        # A218 — Cluster skills into competencies periodically (every 5 steps)
        self._step_count += 1
        if hasattr(self.bridge, 'skills') and hasattr(self.bridge, 'competencies'):
            try:
                if self.bridge.skills.get_skill_count() > 0 and self._step_count % 5 == 0:
                    # Get all skills as (name, vec) pairs
                    skill_pairs = [(s.get("name", ""), s.get("vec")) for s in self.bridge.skills.skills if s.get("vec") is not None]
                    
                    if skill_pairs:
                        # Cluster skills
                        self.bridge.competencies.cluster_skills(skill_pairs)
                        
                        # Log competency update
                        competency_status = self.bridge.competencies.status()
                        if hasattr(self.bridge, 'memory_store') and self.bridge.memory_store is not None:
                            try:
                                if hasattr(self.bridge.memory_store, 'log_thought_event'):
                                    self.bridge.memory_store.log_thought_event({
                                        "type": "competency_update",
                                        "clusters": competency_status
                                    })
                            except Exception:
                                pass
                        
                        # Also log via logger if available
                        if hasattr(self.bridge, 'logger'):
                            try:
                                self.bridge.logger.write({
                                    "competency_update": {
                                        "clusters": competency_status,
                                        "cluster_count": self.bridge.competencies.get_cluster_count()
                                    }
                                })
                            except Exception:
                                pass
            except Exception as e:
                # If clustering fails, continue without it
                if hasattr(self.bridge, 'logger'):
                    try:
                        self.bridge.logger.write({"competency_clustering_error": str(e)})
                    except Exception:
                        pass

        self.last_output = {
            "action": action,
            "action_output": action_output,
            "chosen_thought_debug": dbg,
            "recalled_memories": recalled,
            "drift": drift_state,
            "fusion": fusion_state,
            "attention": attention_state,
            "uncertainty": {
                "value": round(self._uncertainty_value, 4),
                "level": "low" if self._uncertainty_value < 0.25 else "moderate" if self._uncertainty_value < 0.5 else "high" if self._uncertainty_value < 0.75 else "extreme"
            } if hasattr(self, '_uncertainty_value') else {"value": None, "level": None},  # A216 — Uncertainty signal
            "active_task": next_task if next_task else None,
            "evolution_trend": trend,
            "trajectory_vector_preview": evo_vec[:8].tolist() if evo_vec is not None and hasattr(evo_vec, '__getitem__') and len(evo_vec) > 8 else None,
            "cognitive_mode": mode_early,
            "personality_drift": getattr(self, '_personality_drift_info', None),
            "personality_gradient": {
                "active": self.bridge.personality_gradient.gradient_vector is not None,
                "window": len(self.bridge.personality_gradient.history)
            },
            "personality_signature": getattr(self, '_personality_signature_info', {"active": False}),
            "continuity": self.bridge.personality_continuity.summary(),
            "workspace_continuity": getattr(self, '_continuity_vec', None),
            "active_goals": getattr(self, '_goal_summary', []),  # A203 — Emergent goals
            "goal_modulation": self.bridge.goal_modulator.summary(getattr(self, '_goal_modulation', None)),  # A204 — Goal modulation
            "fabricated_goal": self.bridge.goal_fabricator.summary(getattr(self, '_fabricated_goal', None)) if hasattr(self.bridge, 'goal_fabricator') else None,  # A205 — Fabricated goal
            "harmonized_goal": self.bridge.goal_harmonizer.summary(self.bridge.last_harmonized_goal) if hasattr(self.bridge, 'goal_harmonizer') and hasattr(self.bridge, 'last_harmonized_goal') else None,  # A206 — Harmonized goal
            "path_shaping": self.bridge.path_shaper.summary(getattr(self, '_path_state', None)) if hasattr(self.bridge, 'path_shaper') else None,  # A207 — Path shaping
            "subgoal_generator": self.bridge.subgoal_generator.summary(getattr(self, '_subgoal_state', None)) if hasattr(self.bridge, 'subgoal_generator') else None,  # A208 — Subgoal generator
            "subgoal_competition": self.bridge.subgoal_competition.summary(getattr(self, '_competition_state', None)) if hasattr(self.bridge, 'subgoal_competition') else None,  # A209 — Subgoal competition
            "subgoal_routing": self.bridge.subgoal_router.summary(getattr(self, '_routing_state', None)) if hasattr(self.bridge, 'subgoal_router') else None,  # A210 — Subgoal routing
            "sequential_planning": self.bridge.planning_engine.summary(getattr(self, '_planning_state', None)) if hasattr(self.bridge, 'planning_engine') else None,  # A211 — Sequential planning
            "multi_step_update": getattr(self, '_chain_update', None),  # A212 — Chain monitoring update
            "current_chain": [sg.get("id") for sg in self.active_chain] if self.active_chain else None,  # A212 — Current active chain
            "chain_step_index": self.chain_step_index,  # A212 — Current step in chain
            "chain_memory": self.bridge.chain_memory.summary() if hasattr(self.bridge, 'chain_memory') else None,  # A213 — Chain memory
            "skill_encoder": self.bridge.skill_encoder.summary() if hasattr(self.bridge, 'skill_encoder') else None,  # A214 — Skill encoder
            "skill_generalizer": self.bridge.skill_generalizer.summary() if hasattr(self.bridge, 'skill_generalizer') else None,  # A215 — Skill generalization
            "skill_manager": self.bridge.skills.status() if hasattr(self.bridge, 'skills') else None,  # A217 — Skill manager
            "competency_manager": self.bridge.competencies.status() if hasattr(self.bridge, 'competencies') else None,  # A218 — Competency clustering
            "competency_activation": {
                "activations": [(name, round(sim, 4)) for name, sim in getattr(self, '_competency_activations', [])],
                "bias": round(getattr(self, '_competency_bias', 0.0), 4),
                "count": len(getattr(self, '_competency_activations', []))
            } if hasattr(self, '_competency_activations') else None,  # A219 — Competency activation
            "competency_synergy": {
                "edges": [(A, B, round(sim, 4)) for A, B, sim in getattr(self, '_synergy_edges', [])],
                "bonus": round(getattr(self, '_synergy_bonus', 0.0), 4),
                "edge_count": len(getattr(self, '_synergy_edges', []))
            } if hasattr(self, '_synergy_edges') else None,  # A220 — Competency synergy
            "thought_signature": {
                "preview": getattr(self, '_signature_preview', None),
                "synergy_bonus": round(getattr(self, '_synergy_bonus', 0.0), 4),
                "active": hasattr(self.bridge, 'thought_signature') and self.bridge.thought_signature is not None
            } if hasattr(self, '_signature_preview') else {"active": False},  # A221 — Thought Signature
            "personality_flow_field": self.bridge.flow.debug_status() if hasattr(self.bridge, 'flow') and self.bridge.flow is not None else {"active": False},  # A223 — Personality Flow Field
            "self_model": self.bridge.self_model.summary(),
        }

        # Save persistent memory items
        self.bridge.memory_store.log_thought_event(dbg)
        self.bridge.memory_store.log_memory_recall(recalled)
        self.bridge.memory_store.log_drift(drift_state)
        
        if action == "generate_reflection":
            self.bridge.memory_store.log_reflection(action_output)
        
        if action == "update_identity":
            self.bridge.memory_store.log_identity_update(action_output)
        
        if action == "enter_adaptive_evolution" and action_output.get("activated"):
            # Log the evolutionary moment - this is a critical memory imprint
            self.bridge.memory_store.log_thought_event({
                "type": "evolution_activation",
                "cycle": action_output.get("cycle"),
                "reason": action_output.get("reason"),
                "timestamp": time.time()
            })
            self.bridge.logger.write({
                "event": "evolution_activated",
                "cycle": action_output.get("cycle"),
                "reason": action_output.get("reason")
            })

        # Write runtime entry
        self.bridge.logger.write(self.last_output)
        
        # A172: Broadcast cognitive step into the conscious workspace
        try:
            # Get evolution vector from trajectory if available
            evolution_vec = None
            if hasattr(self, 'last_output') and self.last_output.get("evolution_trajectory"):
                trajectory = self.last_output.get("evolution_trajectory", {})
                evolution_vec = trajectory.get("vector")
            
            self.bridge.workspace.broadcast(
                current_thought=chosen_embedding,
                current_action=action,
                memory_recall=recalled,
                identity_state=self.bridge.state.timescales.identity_vector,
                reflection=action_output if action == "generate_reflection" else None,
                attention_focus=attention_state,
                fusion_state=fusion_state,
                task_context=next_task,
                evolution_vector=evolution_vec,
            )
            
            # Add workspace snapshot to output
            workspace_snapshot = self.bridge.workspace.snapshot()
            self.last_output["workspace"] = workspace_snapshot
            
            # A173 — Log spotlight state
            self.bridge.logger.write({"workspace_spotlight": workspace_snapshot})
        except Exception as e:
            # If workspace broadcast fails, continue without it
            if hasattr(self.bridge, 'logger'):
                self.bridge.logger.write({"workspace_broadcast_error": str(e)})
        
        # Log task engagement if task is active
        if next_task:
            self.bridge.logger.write({"engaged_task": next_task})

        # Update stability engine
        reflection = action_output if action == "generate_reflection" else None
        semantic_top = recalled[0] if recalled else None

        stability = self.bridge.stability.update(
            drift_state.get("latest_drift"),
            self.bridge.fusion.last_fusion_vector,
            self.bridge.state.timescales.identity_vector,
            reflection,
            semantic_top
        )

        self.bridge.ready_for_adaptive_evolution = stability["ready_for_adaptive_evolution"]
        self.bridge.stability_report = stability
        self.last_output["stability"] = stability

        # A157: Attempt evolution mutation if enabled
        if self.bridge.evolution.active and self.bridge.evolution.evolution_enabled:
            # Convert stability report to format expected by mutation system
            stability_for_mutation = {
                "stable": stability.get("ready_for_adaptive_evolution", False),
                "stable_for": stability.get("stable_for_cycles", 0)
            }
            drift_value = drift_state.get("latest_drift") or 0.0
            coherence_value = fusion_state.get("coherence", 1.0) if isinstance(fusion_state, dict) else 1.0
            
            evo_outcome = self.bridge.evolution.attempt_mutation(
                stability_for_mutation,
                drift_value,
                coherence_value
            )
            self.last_output["evolution_mutation"] = evo_outcome
            
            # A159: Feed evolution event into consolidation buffer
            if evo_outcome:
                self.bridge.evo_consolidator.record(evo_outcome)
            
            # A159: Run consolidation pass
            consolidation_result = self.bridge.evo_consolidator.consolidate(
                self.bridge.state.memory_manager if hasattr(self.bridge.state, "memory_manager") else None,
                self.bridge.hooks,
                self.bridge.state.timescales
            )
            if consolidation_result:
                self.last_output["evolution_consolidation"] = consolidation_result
                
                # A160: Feed consolidated evolution into predictor
                self.bridge.evo_predictor.record(consolidation_result)
            
            # A160: Trajectory prediction (update after evolution mutations)
            # Use updated trajectory if available, otherwise recompute
            trajectory_updated = self.bridge.evo_predictor.predict(
                drift_state,
                self.bridge.state.timescales,
                self.bridge.hooks
            )
            # Use updated trajectory if it exists, otherwise keep early one
            if trajectory_updated:
                trajectory = trajectory_updated
                trend = trajectory.get("trend", "neutral")
                evo_vec = trajectory.get("vector")
            
            self.last_output["evolution_trajectory"] = trajectory
            
            # A161: Recalibrate attention using trajectory
            self.bridge.attention.recalibrate_with_evolution(trajectory)
            
            # A162: Apply evolution to Fusion Engine
            # ---------------------------------------------
            # A163 — Feed evolution into Fusion Engine (also done here for completeness)
            # ---------------------------------------------
            try:
                drift_value = drift_state.get("latest_drift") or 0.0
                self.bridge.fusion.apply_evolutionary_adjustment(
                    trajectory=trajectory,
                    drift_level=drift_value
                )
            except Exception as e:
                # Log error but don't break the cognitive cycle
                if hasattr(self.bridge, 'logger'):
                    self.bridge.logger.write({"fusion_evo_error": str(e)})
            
            # A158: Evolutionary Reflection Integration
            if evo_outcome and (
                evo_outcome.get("evolved") or
                evo_outcome.get("rolled_back")
            ):
                # Construct reflection text
                try:
                    if evo_outcome.get("evolved"):
                        old_val = evo_outcome.get('old', 0)
                        new_val = evo_outcome.get('new', 0)
                        if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                            text = (
                                f"I evolved internally by adjusting a cognitive parameter "
                                f"from {old_val:.4f} to {new_val:.4f}. "
                                "This modification improved coherence."
                            )
                        else:
                            text = (
                                f"I evolved internally by adjusting a cognitive parameter. "
                                "This modification improved coherence."
                            )
                    else:
                        old_val = evo_outcome.get('old', 0)
                        new_attempt = evo_outcome.get('new_attempt', 0)
                        if isinstance(old_val, (int, float)) and isinstance(new_attempt, (int, float)):
                            text = (
                                f"I attempted a cognitive evolution from {old_val:.4f} "
                                f"to {new_attempt:.4f} but reverted the change "
                                "to maintain stability."
                            )
                        else:
                            text = (
                                "I attempted a cognitive evolution but reverted the change "
                                "to maintain stability."
                            )
                except Exception as e:
                    # Fallback if formatting fails
                    text = (
                        "I engaged in internal cognitive evolution. "
                        "The change was evaluated and handled appropriately."
                    )

                # Encode reflection
                reflection_vec = self.bridge.hooks.on_reflection(text)

                # Store in semantic memory
                if hasattr(self.bridge.state, "memory_manager") and self.bridge.state.memory_manager:
                    import time
                    concept_name = f"evolution_reflection_{int(time.time())}"
                    self.bridge.state.memory_manager.store_concept(concept_name, reflection_vec)
                    
                    # Also log to memory store
                    self.bridge.memory_store.log_thought_event({
                        "type": "evolution_reflection",
                        "text": text,
                        "evolved": evo_outcome.get("evolved", False),
                        "timestamp": time.time()
                    })
                    
                    # Store in output for visibility
                    self.last_output["evolution_reflection"] = {
                        "text": text,
                        "concept_name": concept_name
                    }

        # If evolution active, embed into output
        self.last_output["evolution_status"] = self.bridge.evolution.status()

        return self.last_output

    def status(self):
        return self.last_output

