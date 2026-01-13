// src/phase/phase_descriptor.ts
//
// ⚠️ PHASE I — READ-ONLY DATA STRUCTURE ONLY
// This file defines a passive phase descriptor structure.
// It must NOT be imported into cognition, action selection, or memory paths.
//
// 🔒 PHASE I CONSTRAINT: No setters, no mutation logic, no logic that "chooses" a phase.
// This is descriptive metadata only.

/**
 * Phase Descriptor — Passive metadata structure
 * 
 * Represents runtime phase metadata for observational purposes only.
 * This structure is read-only and does not influence behavior.
 */
export interface PhaseDescriptor {
  /** Name of the current phase */
  phase_name: string;
  
  /** Timestamp when phase was entered (milliseconds since epoch) */
  entered_at: number;
  
  /** Timestamp when phase was last evaluated (milliseconds since epoch) */
  last_evaluated_at: number;
  
  /** Authority source that declared this phase */
  authority: string;
  
  /** Entity that declared this phase */
  declared_by: string;
  
  /** Confidence level (0.0 to 1.0) */
  confidence: number;
}

/**
 * Creates a default Phase Descriptor for QUIET_WAKE phase.
 * 
 * ⚠️ PHASE I CONSTRAINT: This is a factory function only.
 * It does not choose phases or influence behavior.
 * 
 * @returns Default PhaseDescriptor with QUIET_WAKE defaults
 */
export function createDefaultPhaseDescriptor(): PhaseDescriptor {
  const now = Date.now();
  return {
    phase_name: "QUIET_WAKE",
    entered_at: now,
    last_evaluated_at: now,
    authority: "implicit",
    declared_by: "runtime",
    confidence: 1.0
  };
}

/**
 * Creates a Phase Descriptor with explicit values.
 * 
 * ⚠️ PHASE I CONSTRAINT: This is a factory function only.
 * It does not validate, choose, or influence phases.
 * 
 * @param phase_name - Name of the phase
 * @param authority - Authority source (default: "implicit")
 * @param declared_by - Declaring entity (default: "runtime")
 * @param confidence - Confidence level (default: 1.0)
 * @returns PhaseDescriptor with specified values
 */
export function createPhaseDescriptor(
  phase_name: string,
  authority: string = "implicit",
  declared_by: string = "runtime",
  confidence: number = 1.0
): PhaseDescriptor {
  const now = Date.now();
  return {
    phase_name,
    entered_at: now,
    last_evaluated_at: now,
    authority,
    declared_by,
    confidence
  };
}

// ⚠️ PHASE I INVARIANT GUARD
// This module must never be imported by:
// - Action engine
// - Arbitration logic
// - Evolution / learning modules
// - Any code that makes behavioral decisions
//
// If you see this module imported in a behavioral path, STOP and report.
