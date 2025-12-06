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

        # If evolution active, embed into output
        self.last_output["evolution_status"] = self.bridge.evolution.status()

        return self.last_output

    def status(self):
        return self.last_output

