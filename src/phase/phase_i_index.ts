// src/phase/phase_i_index.ts
//
// ⚠️ PHASE I — SCAFFOLDING INDEX
// This file serves as the Phase I scaffolding entry point.
// All Phase I modules are read-only and telemetry-only.
//
// 🔒 PHASE I INVARIANT GUARDS:
// - No behavioral changes
// - No action blocking
// - No phase enforcement
// - Read-only observation
// - Write-only telemetry

export { 
  PhaseDescriptor,
  createDefaultPhaseDescriptor,
  createPhaseDescriptor
} from "./phase_descriptor.ts";

export {
  PhaseObserver,
  PhaseObserverInput,
  PhaseObserverOutput,
  phaseObserver
} from "./phase_observer.ts";

export {
  PhaseTelemetryMetadata,
  createPhaseTelemetryMetadata,
  emitPhaseTelemetry,
  extendTelemetryWithPhase
} from "./phase_telemetry.ts";

// ⚠️ PHASE I ENFORCEMENT BOUNDARY
// This module must never be imported by:
// - src/action_layer/action_engine.ts
// - src/dual_core/arbitration_contract.ts
// - Any evolution / learning modules
// - Any code that makes behavioral decisions
//
// Phase I is structural + observational only.
// If you see behavioral code importing this, STOP and report.
