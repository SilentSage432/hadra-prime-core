"""
HADRA-PRIME — Autonomous Runtime Loop (A150)
--------------------------------------------
This script activates PRIME's continuous cognitive loop.

The loop:
1. Initializes PRIME's neural bridge
2. Performs endless cognitive steps
3. Logs internal states for monitoring
4. Runs until manually stopped by operator

This is the moment PRIME becomes a continuous cognitive process.
"""

import time
import traceback

from src.neural.neural_bridge import NeuralBridge
from src.cognition.emergent.concept_observer import ConceptObserver


class PrimeRuntime:

    def __init__(self, loop_interval=0.35):
        self.loop_interval = loop_interval
        self.bridge = NeuralBridge()
        self.running = True
        self._last_output = None
        self.concept_observer = ConceptObserver()

    def start(self):
        print("🔥 HADRA-PRIME cognitive runtime started")
        print("Press CTRL+C to stop.\n")

        while self.running:
            try:
                # Execute a single cognitive cycle
                output = self.bridge.cognitive_step()
                # Store last output for observer access (read-only)
                self._last_output = output

                # Observe fusion vector for concept formation
                try:
                    if hasattr(self.bridge, 'fusion') and hasattr(self.bridge.fusion, 'last_fusion_vector'):
                        fusion_vector = self.bridge.fusion.last_fusion_vector
                        if fusion_vector is not None:
                            self.concept_observer.observe(fusion_vector)
                except Exception:
                    # ECFL cannot stop ADRAE - silent failure
                    pass

                # Log (you can later redirect this to a file)
                print("—— Cognitive Step ——")
                print(f"Action: {output['action']}")
                print(f"Thought Debug: {output['chosen_thought_debug']}")
                print(f"Recalled: {output['recalled_memories']}")
                print(f"Drift: {output['drift']}")
                print(f"Fusion: {output['fusion']}")
                print(f"Attention: {output['attention']}")
                print()

                # Control loop pacing
                time.sleep(self.loop_interval)

            except KeyboardInterrupt:
                print("\n🛑 PRIME runtime halted by operator.")
                self.running = False

            except Exception as e:
                print("\n❌ ERROR IN RUNTIME LOOP ❌")
                traceback.print_exc()
                # Continue running unless the operator halts it manually
                time.sleep(self.loop_interval)


if __name__ == "__main__":
    from threading import Thread
    from src.prime_core.observable_state import export_observable_state
    from src.observers.minimal_observer import observer_loop

    runtime = PrimeRuntime(loop_interval=0.35)

    def read_state():
        return export_observable_state(runtime)

    observer_thread = Thread(
        target=observer_loop,
        args=(read_state,),
        daemon=True
    )

    observer_thread.start()
    runtime.start()

