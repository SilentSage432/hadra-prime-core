// src/cognition/proto_goal_engine.ts

import { SEL } from "../emotion/sel.ts";
import { MotivationEngine, type MotivationVector } from "./motivation_engine.ts";
import { StabilityMatrix } from "../stability/stability_matrix.ts";

export interface ProtoGoal {
  type: string;
  priority: number; // 0–1 scaled priority
  reason: string;
}

export class ProtoGoalEngine {
  static computeGoals(): ProtoGoal[] {
    const sel = SEL.getState();
    const stability = StabilityMatrix.getSnapshot();
    const motivation: MotivationVector = MotivationEngine.compute();

    const goals: ProtoGoal[] = [];

    // -----------------------------------------
    // 1. Clarity Seeking Goal
    // -----------------------------------------
    if (motivation.claritySeeking > 0.4) {
      goals.push({
        type: "increase_clarity",
        priority: motivation.claritySeeking,
        reason: "High clarity-seeking pressure from SEL + motivation vector",
      });
    }

    // -----------------------------------------
    // 2. Curiosity / Exploration Goal
    // -----------------------------------------
    if (motivation.curiosity > 0.35) {
      goals.push({
        type: "explore_context",
        priority: motivation.curiosity,
        reason: "Low certainty combined with curiosity pressure",
      });
    }

    // -----------------------------------------
    // 3. Diagnostics Goal
    // -----------------------------------------
    if (motivation.stabilityPressure > 0.45) {
      goals.push({
        type: "run_micro_diagnostics",
        priority: motivation.stabilityPressure,
        reason: "StabilityMatrix pressure indicates potential issues",
      });
    }

    // -----------------------------------------
    // 4. Consolidation Goal
    // -----------------------------------------
    if (motivation.consolidation > 0.4) {
      goals.push({
        type: "consolidate_memory",
        priority: motivation.consolidation,
        reason: "High consolidation pressure from SEL coherence",
      });
    }

    // -----------------------------------------
    // 5. Emotional Baseline Goal
    // -----------------------------------------
    if (sel.tension > 0.45 || sel.valence < 0.3) {
      goals.push({
        type: "stabilize_emotional_state",
        priority: sel.tension,
        reason: "Emotional tension indicates need for SEL baseline correction",
      });
    }

    // -----------------------------------------
    // 6. Operator Input Goal
    // -----------------------------------------
    if (motivation.urgency > 0.55 && motivation.claritySeeking > 0.3) {
      goals.push({
        type: "request_operator_input",
        priority: motivation.urgency,
        reason: "High urgency combined with unclear intent direction",
      });
    }

    // -----------------------------------------
    // 7. Federation Integrity Goal
    // -----------------------------------------
    const instability = stability ? (1 - (stability.score || 0.5)) : 0;
    if (instability > 0.4) {
      goals.push({
        type: "protect_federation_integrity",
        priority: instability,
        reason: "System instability indicates potential threat to coherence",
      });
    }

    // -----------------------------------------
    // Sort by priority (highest first)
    // -----------------------------------------
    goals.sort((a, b) => b.priority - a.priority);

    return goals;
  }
}

