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
from src.cognition.observation_ledger import ObservationLedger
from src.cognition.internal_events import get_event_logger
from src.cognition.inference_cooldown import InferenceTracker
from persistence.log_writer import LogWriter
# ⚠️ PHASE II: Rhythm Observer (read-only, append-only)
from src.observers.rhythm_observer import get_rhythm_observer
# ⚠️ PHASE I: Phase Observer (read-only, append-only, Python-native)
from src.observers.phase_observer import get_phase_observer, PhaseTelemetryEmitter
# ⚠️ PHASE III: Rotation Baselines & Window Detection (read-only, append-only)
from src.telemetry.phase3_rotation_observer import get_phase3_rotation_observer


class PrimeRuntime:

    def __init__(self, loop_interval=0.35):
        self.loop_interval = loop_interval
        self.bridge = NeuralBridge()
        self.running = True
        self._last_output = None
        self.concept_observer = ConceptObserver()
        self.observation_ledger = ObservationLedger()
        self.inference_tracker = InferenceTracker()
        self.event_logger = get_event_logger()
        # Shared log writer sink (prime_runtime.log)
        self.log_writer = LogWriter()
        # ⚠️ PHASE III: Initialize rotation observer (read-only, append-only)
        self.phase3_observer = get_phase3_rotation_observer()
        # ⚠️ PHASE II: Initialize rhythm observer (read-only, append-only)
        # Pass Phase III observer to Phase II for preferred payload hook
        self.rhythm_observer = get_rhythm_observer(phase3_observer=self.phase3_observer)
        # ⚠️ PHASE I: Initialize phase observer (read-only, append-only, Python-native)
        self.phase_observer = get_phase_observer()

    def start(self):
        print("🔥 HADRA-PRIME cognitive runtime started")
        print("Press CTRL+C to stop.\n")
        # Diagnostic: Verify observers initialized
        print(f"[DIAGNOSTIC] Phase I observer initialized: {self.phase_observer is not None}", flush=True)
        print(f"[DIAGNOSTIC] Phase II observer initialized: {self.rhythm_observer is not None}", flush=True)
        print(f"[DIAGNOSTIC] Phase I last_observation_time: {self.phase_observer.last_observation_time}", flush=True)
        print(f"[DIAGNOSTIC] Phase II last_observation_time: {self.rhythm_observer.last_observation_time}", flush=True)

        while self.running:
            try:
                # Execute a single cognitive cycle
                output = self.bridge.cognitive_step()
                # Store last output for observer access (read-only)
                self._last_output = output
                
                # ⚠️ PHASE II: Observe rhythm (read-only, does not modify output or delay loop)
                # This checks if 60 seconds have passed and emits [ADRAE-RHYTHM] logs if needed
                # Phase II will also feed payload to Phase III observer (preferred)
                try:
                    self.rhythm_observer.observe_step(output, self.log_writer)
                except Exception as e:
                    # Log error but don't stop the loop
                    print(f"[PHASE-II-ERROR] Rhythm observer failed: {e}", flush=True)
                
                # ⚠️ PHASE III: Also observe per-step (for action histogram tracking)
                # Primary input is via Phase II rhythm payload (above), but we also observe steps
                try:
                    self.phase3_observer.observe_step(output)
                    # Periodic tick (internally rate-limited, safe to call every step)
                    current_time = time.monotonic()
                    self.phase3_observer.tick(current_time)
                except Exception as e:
                    # Log error but don't stop the loop
                    print(f"[PHASE-III-ERROR] Rotation observer failed: {e}", flush=True)
                
                # ⚠️ PHASE I: Observe phase (read-only, gated by 60-second cadence, no threads/no loops)
                # Check if 60 seconds have passed since last Phase I observation
                # OR if this is the first observation (current_phase is None)
                try:
                    current_time = time.monotonic()
                    should_check = (
                        self.phase_observer.should_emit(current_time) or 
                        self.phase_observer.current_phase is None
                    )
                    
                    if should_check:
                        # Observe runtime state (read-only, no behavior change)
                        observer_output = self.phase_observer.observe(output, self.bridge)
                        
                        # Emit log ONLY when phase changes (not every interval)
                        if observer_output.get("phase_changed", False):
                            PhaseTelemetryEmitter.emit_phase_change(
                                observer_output["phase"],
                                observer_output["metrics"],
                                observer_output["confidence"],
                                self.log_writer
                            )
                        
                        # Update observation time (internal tracking only)
                        self.phase_observer.update_observation_time(current_time)
                except Exception as e:
                    # Log error but don't stop the loop
                    print(f"[PHASE-I-ERROR] Phase observer failed: {e}", flush=True)
                    import traceback
                    traceback.print_exc()

                # Record internal inference as observation
                try:
                    fusion_vector = None
                    if hasattr(self.bridge, 'fusion') and hasattr(self.bridge.fusion, 'last_fusion_vector'):
                        fusion_vector = self.bridge.fusion.last_fusion_vector
                    
                    if fusion_vector is not None:
                        # Record fusion output as internal inference observation
                        fusion_hash = self.observation_ledger.record(
                            source="internal",
                            channel="signal",
                            payload={
                                "type": "fusion_output",
                                "dim": fusion_vector.numel() if hasattr(fusion_vector, 'numel') else len(fusion_vector)
                            },
                            metadata={
                                "provenance": {
                                    "origin": "internal_inference",
                                    "inputs": [],  # Will be populated if tracking input sources
                                    "confidence": 0.5,  # Default confidence, can be refined
                                    "revisable": True
                                }
                            }
                        )
                        
                        # Track inference for cooldown window
                        inference_id = f"fusion_{fusion_hash}"
                        count = self.inference_tracker.record_observation(inference_id)
                        
                        # Check if inference is ready (passed cooldown)
                        if count == self.inference_tracker.cooldown_threshold:
                            self.event_logger.log_event(
                                "inference_stabilized",
                                {"inference_id": inference_id, "observation_count": count}
                            )
                        
                        # Observe fusion vector for concept formation
                        self.concept_observer.observe(fusion_vector)
                        
                        # Check for pattern density (repeated observations)
                        if count > 1:
                            self.event_logger.log_event(
                                "observation_repeat",
                                {"inference_id": inference_id, "count": count}
                            )
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

