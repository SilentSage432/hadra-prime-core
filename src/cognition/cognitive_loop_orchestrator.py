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

                # Recompute attention with updated timescales
                if hasattr(self.bridge, "attention"):
                    self.bridge.attention.compute_attention_vector(self.bridge.state.timescales)

                # Recompute fusion with updated attention and timescales
                if hasattr(self.bridge, "fusion"):
                    self.bridge.fusion.fuse(
                        self.bridge.attention.last_focus_vector,
                        self.bridge.state.timescales
                    )
                    
                    # === A165: Personality Drift Regulation ===
                    identity_vec = self.bridge.state.timescales.identity_vector
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
        # A163 — Evolution-biased cognitive action selection
        # ---------------------------------------------
        action = self.bridge.choose_cognitive_action()
        
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

