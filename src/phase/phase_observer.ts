// src/phase/phase_observer.ts
//
// ⚠️ PHASE I — READ-ONLY OBSERVER ONLY
// This module observes runtime state and suggests a phase.
// It does NOT enforce, block, or influence behavior.
//
// 🔒 PHASE I CONSTRAINT: 
// - No thresholds that change behavior
// - No branching that blocks actions
// - No writing to state
// - No imports into action engine

import { PhaseDescriptor, createDefaultPhaseDescriptor } from "./phase_descriptor.ts";

/**
 * Observer input snapshot (read-only)
 * 
 * Contains only read-only observations of runtime state.
 */
export interface PhaseObserverInput {
  /** Last executed action name (if available) */
  last_action_name?: string;
  
  /** Drift metrics (already computed, read-only) */
  drift_metrics?: {
    concept_drift?: number;
    prediction_variance?: number;
    emotional_drift?: number;
  };
  
  /** Guardian signal snapshots (read-only, no rewiring) */
  guardian_signals?: Array<{
    guardian_id: string;
    signal_type: "VETO" | "ESCALATION" | "WARNING" | "INFO";
    severity?: number;
  }>;
  
  /** External input presence (boolean only) */
  external_input_present?: boolean;
}

/**
 * Observer output (suggestion only)
 * 
 * This is a suggestion, not an enforcement.
 */
export interface PhaseObserverOutput {
  /** Suggested phase name */
  phase: string;
  
  /** Confidence in suggestion (0.0 to 1.0) */
  confidence: number;
}

/**
 * Phase Observer — Purely Observational
 * 
 * Observes runtime state and suggests a phase based on current conditions.
 * This observer does NOT:
 * - Enforce phase transitions
 * - Block actions
 * - Modify state
 * - Influence behavior
 * 
 * ⚠️ PHASE I CONSTRAINT: This is read-only observation only.
 */
export class PhaseObserver {
  private observerVersion: string = "1.0.0-phase-i";
  
  /**
   * Observe runtime state and suggest a phase.
   * 
   * This method is purely observational and does not influence behavior.
   * 
   * @param input - Read-only snapshot of runtime state
   * @returns Suggested phase with confidence level
   */
  observe(input: PhaseObserverInput): PhaseObserverOutput {
    // ⚠️ PHASE I CONSTRAINT: This logic is descriptive only.
    // It does not enforce, block, or gate any actions.
    
    // Default to QUIET_WAKE (the natural state)
    let suggestedPhase = "QUIET_WAKE";
    let confidence = 0.98;
    
    // Read-only observation: Check for guardian veto signals
    // This does NOT block actions, only suggests phase
    if (input.guardian_signals) {
      const vetoPresent = input.guardian_signals.some(
        signal => signal.signal_type === "VETO"
      );
      if (vetoPresent) {
        // Observer suggests remaining in current phase when veto present
        // This is descriptive only, not enforcement
        suggestedPhase = "QUIET_WAKE";
        confidence = 0.95;
      }
    }
    
    // Read-only observation: Check drift metrics
    // This does NOT change behavior, only suggests phase
    if (input.drift_metrics) {
      const hasSignificantDrift = 
        (input.drift_metrics.concept_drift && Math.abs(input.drift_metrics.concept_drift) > 0.3) ||
        (input.drift_metrics.emotional_drift && Math.abs(input.drift_metrics.emotional_drift) > 0.3);
      
      if (hasSignificantDrift) {
        // Observer suggests QUIET_WAKE when drift is significant
        // This is descriptive only, not enforcement
        suggestedPhase = "QUIET_WAKE";
        confidence = 0.92;
      }
    }
    
    // Read-only observation: Check external input
    // This does NOT gate actions, only suggests phase
    if (input.external_input_present) {
      // Observer may suggest different phase if external input present
      // For Phase I, we default to QUIET_WAKE
      suggestedPhase = "QUIET_WAKE";
      confidence = 0.90;
    }
    
    return {
      phase: suggestedPhase,
      confidence
    };
  }
  
  /**
   * Get observer version for telemetry.
   * 
   * @returns Observer version string
   */
  getVersion(): string {
    return this.observerVersion;
  }
}

// Singleton instance (read-only observer)
export const phaseObserver = new PhaseObserver();

// ⚠️ PHASE I INVARIANT GUARD
// This module must never be imported by:
// - Action engine
// - Arbitration logic
// - Evolution / learning modules
// - Any code that makes behavioral decisions
//
// This observer only suggests phases. It does not enforce them.
// If you see this module used to block or gate actions, STOP and report.
