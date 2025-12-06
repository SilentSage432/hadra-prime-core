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
        """

        # Increment cycle counter
        self.bridge.cycle_count += 1

        # 0. Retrieve next task (if any)
        next_task = self.bridge.tasks.peek()
        if next_task:
            # Task becomes part of debug data
            task_embedding = self.bridge.hooks.on_perception(next_task["text"])
        else:
            task_embedding = None

        # 1. Propose candidate thoughts
        candidates = self.bridge.propose_thoughts()
        
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
            except Exception as e:
                print("🔥 ERROR: Failed to update cognition state:", e)

        # 3. Execute a cognitive action
        action = self.bridge.choose_cognitive_action()
        action_output = self.bridge.perform_action(action)

        # 4. Memory metabolism
        recalled = self.bridge.memory_cycle()

        # 5. Update attention & fusion (internal to bridge)
        # Fusion gets recomputed automatically on next perception/reflection

        # 6. Drift & coherence awareness
        drift_state = self.bridge.state.drift.get_status()
        fusion_state = self.bridge.fusion.status()
        attention_state = self.bridge.attention.status()

        self.last_output = {
            "action": action,
            "action_output": action_output,
            "chosen_thought_debug": dbg,
            "recalled_memories": recalled,
            "drift": drift_state,
            "fusion": fusion_state,
            "attention": attention_state,
            "active_task": next_task if next_task else None,
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

