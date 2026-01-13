// src/phase/phase_engine.ts
//
// ⚠️ PHASE I — BEHAVIORAL INVARIANT
// This engine's behavior must remain unchanged.
// Phase I scaffolding is observational and telemetry-only.

import { PRIME_LOOP_GUARD } from "../shared/loop_guard.ts";
// ⚠️ PHASE I: Import Phase I modules (read-only, telemetry-only)
import { createDefaultPhaseDescriptor } from "./phase_descriptor.ts";
import { phaseObserver, PhaseObserverInput } from "./phase_observer.ts";
import { createPhaseTelemetryMetadata, emitPhaseTelemetry } from "./phase_telemetry.ts";

export async function phaseEngine(state: any) {
  if (!PRIME_LOOP_GUARD.enter()) {
    console.log("[PRIME-PHASE] Blocked recursive entry");
    return state;
  }

  try {
    if (state.intent === null) {
      console.log("[PRIME-PHASE] Null intent — halting pipeline");
      return state;
    }

    if (state.safety?.halt) {
      console.log("[PRIME-PHASE] Safety halt engaged — stopping");
      return state;
    }

    // ⚠️ PHASE I: Observational scaffolding (does not change behavior)
    // Create phase descriptor for telemetry (read-only)
    const phaseDescriptor = createDefaultPhaseDescriptor();
    
    // ⚠️ PHASE I: Observe runtime state (read-only, no behavior change)
    // Build observer input from current state (read-only access only)
    const observerInput: PhaseObserverInput = {
      last_action_name: state.intent?.type || undefined,
      drift_metrics: state.prediction ? {
        concept_drift: state.prediction.conceptDrift,
        prediction_variance: state.prediction.variance,
        emotional_drift: state.prediction.emotionalDrift
      } : undefined,
      // Guardian signals would come from safety layer (read-only snapshot)
      guardian_signals: state.safety?.guardianSignals || undefined,
      external_input_present: state.intent !== null
    };
    
    // ⚠️ PHASE I: Observer suggests phase (descriptive only, not enforced)
    const observerOutput = phaseObserver.observe(observerInput);
    
    // ⚠️ PHASE I: Emit telemetry (append-only, write-only)
    const phaseTelemetry = createPhaseTelemetryMetadata(
      phaseDescriptor,
      observerOutput,
      phaseObserver.getVersion()
    );
    emitPhaseTelemetry(phaseTelemetry);
    
    // ⚠️ PHASE I BEHAVIORAL INVARIANT: Original behavior unchanged
    // The phase engine still returns "complete" as before.
    // Phase I scaffolding does not influence this return value.
    return {
      ...state,
      phase: "complete",
    };
  } finally {
    PRIME_LOOP_GUARD.exit();
  }
}

// ⚠️ PHASE I INVARIANT GUARD
// This engine's behavior must remain identical to pre-Phase I.
// Phase I scaffolding is:
// - Observational only (read-only state access)
// - Telemetry only (write-only logging)
// - No behavioral changes
// - No action blocking
// - No phase enforcement
//
// If this engine's behavior changes, Phase I has failed.
