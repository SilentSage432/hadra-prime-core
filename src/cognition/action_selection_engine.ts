// src/cognition/action_selection_engine.ts

import type { ProtoGoal } from "./proto_goal_engine.ts";
import { SEL } from "../emotion/sel.ts";
import { StabilityMatrix } from "../stability/stability_matrix.ts";
import MemoryBroker from "../memory/index.ts";

export class ActionSelectionEngine {
  static selectAction(goal: ProtoGoal | null): (() => void) | null {
    if (!goal) return null;

    switch (goal.type) {
      case "increase_clarity":
        return this.adjustClarity;

      case "explore_context":
        return this.requestContextExpansion;

      case "run_micro_diagnostics":
        return this.runMicroDiagnostics;

      case "consolidate_memory":
        return this.consolidateMemory;

      case "stabilize_emotional_state":
        return this.stabilizeSEL;

      case "request_operator_input":
        return this.pingOperatorAttention;

      case "protect_federation_integrity":
        return this.reinforceProtectionMode;

      default:
        return null;
    }
  }

  // -----------------------
  // Internal action methods
  // -----------------------

  private static adjustClarity() {
    console.log("[ACTION] adjust_clarity()");
    SEL.nudgeCoherence(+0.02);
  }

  private static requestContextExpansion() {
    console.log("[ACTION] request_context_expansion()");
    // Action will be surfaced to UI in later phases
  }

  private static runMicroDiagnostics() {
    console.log("[ACTION] run_micro_diagnostics()");
    StabilityMatrix.getSnapshot(); // lightweight check only
  }

  private static consolidateMemory() {
    console.log("[ACTION] consolidate_memory()");
    // Memory consolidation - strengthen recent patterns
    // This is a placeholder for future memory consolidation logic
  }

  private static stabilizeSEL() {
    console.log("[ACTION] stabilize_sel()");
    SEL.normalize();
  }

  private static pingOperatorAttention() {
    console.log("[ACTION] ping_operator_attention()");
    // Later: UI notification hook
  }

  private static reinforceProtectionMode() {
    console.log("[ACTION] reinforce_protection_mode()");
    StabilityMatrix.reinforceInvariant();
  }
}

