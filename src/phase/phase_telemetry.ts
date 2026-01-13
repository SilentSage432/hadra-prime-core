// src/phase/phase_telemetry.ts
//
// ⚠️ PHASE I — APPEND-ONLY TELEMETRY ONLY
// This module extends telemetry schema to include phase metadata.
// Telemetry must remain write-only. No parsing or reuse of phase data.
//
// 🔒 PHASE I CONSTRAINT:
// - Telemetry is write-only
// - No parsing or reuse of phase data
// - No increased verbosity beyond metadata
// - No runtime logic reads this data

import { PhaseDescriptor } from "./phase_descriptor.ts";
import { PhaseObserverOutput } from "./phase_observer.ts";

/**
 * Phase telemetry metadata structure
 * 
 * This structure is appended to telemetry logs only.
 * It is never read by runtime logic.
 */
export interface PhaseTelemetryMetadata {
  phase: {
    /** Current phase name */
    current: string;
    
    /** Confidence level (0.0 to 1.0) */
    confidence: number;
    
    /** Authority source */
    authority: string;
    
    /** Observer version */
    observer_version: string;
  };
}

/**
 * Creates phase telemetry metadata from a phase descriptor and observer output.
 * 
 * ⚠️ PHASE I CONSTRAINT: This function only formats data for logging.
 * It does not influence behavior or read telemetry.
 * 
 * @param descriptor - Phase descriptor (read-only)
 * @param observerOutput - Observer output (read-only)
 * @param observerVersion - Observer version string
 * @returns Telemetry metadata object
 */
export function createPhaseTelemetryMetadata(
  descriptor: PhaseDescriptor,
  observerOutput: PhaseObserverOutput,
  observerVersion: string
): PhaseTelemetryMetadata {
  return {
    phase: {
      current: descriptor.phase_name,
      confidence: observerOutput.confidence,
      authority: descriptor.authority,
      observer_version: observerVersion
    }
  };
}

/**
 * Emits phase telemetry metadata to console.
 * 
 * ⚠️ PHASE I CONSTRAINT: This function only writes to telemetry.
 * It does not parse, read, or reuse telemetry data.
 * 
 * @param metadata - Phase telemetry metadata
 */
export function emitPhaseTelemetry(metadata: PhaseTelemetryMetadata): void {
  // ⚠️ PHASE I CONSTRAINT: This is write-only telemetry.
  // No runtime logic should read or parse this output.
  console.log("[ADRAE-PHASE-TELEMETRY]", JSON.stringify(metadata));
}

/**
 * Extends existing telemetry object with phase metadata.
 * 
 * ⚠️ PHASE I CONSTRAINT: This function only appends metadata.
 * It does not modify existing telemetry logic or behavior.
 * 
 * @param existingTelemetry - Existing telemetry object (read-only)
 * @param phaseMetadata - Phase metadata to append
 * @returns Extended telemetry object (for logging only)
 */
export function extendTelemetryWithPhase(
  existingTelemetry: any,
  phaseMetadata: PhaseTelemetryMetadata
): any {
  // ⚠️ PHASE I CONSTRAINT: This is append-only.
  // We create a new object to avoid mutating existing telemetry.
  return {
    ...existingTelemetry,
    ...phaseMetadata
  };
}

// ⚠️ PHASE I INVARIANT GUARD
// This module must never be imported by:
// - Action engine
// - Arbitration logic
// - Evolution / learning modules
// - Any code that reads or parses telemetry for logic
//
// Telemetry is write-only. If you see this module's data being read
// for behavioral decisions, STOP and report.
