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
        except Exception as e:
            # If goal generation fails, continue without it
            if hasattr(self.bridge, 'logger'):
                self.bridge.logger.write({"goal_generation_error": str(e)})
            self._goal_summary = []
            self._goal_modulation = None
            self._fabricated_goal = None
        
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

        # 2. Select the strongest thought
        if not candidates:
            chosen_embedding = None
            dbg = {"note": "No candidates generated"}
        else:
            result = self.bridge.select_thought(candidates)
            
            # Handle both tuple and single return values
            if isinstance(result, tuple):
                chosen_embedding, dbg = result
                if dbg is None:
                    dbg = {"note": "No debug info available"}
            else:
                chosen_embedding = result
                dbg = {"note": "No debug info available"}

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

        # 6. Drift & coherence awareness
        drift_state = self.bridge.state.drift.get_status()
        fusion_state = self.bridge.fusion.status()
        attention_state = self.bridge.attention.status()
        
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

        self.last_output = {
            "action": action,
            "action_output": action_output,
            "chosen_thought_debug": dbg,
            "recalled_memories": recalled,
            "drift": drift_state,
            "fusion": fusion_state,
            "attention": attention_state,
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

