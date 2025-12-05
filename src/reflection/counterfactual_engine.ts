// src/reflection/counterfactual_engine.ts

import { SEL } from "../emotion/sel.ts";
import { StabilityMatrix } from "../stability/stability_matrix.ts";
import { MotivationEngine } from "../cognition/motivation_engine.ts";
import type { InternalPlan } from "../cognition/planning_engine.ts";

export interface CounterfactualResult {
  predictedOutcome: "stabilizing" | "risky" | "neutral";
  counterfactualScore: number; // 0–1
  deltaTension: number;
  deltaStability: number;
  deltaMotivationUrgency: number;
}

export class CounterfactualEngine {
  /**
   * Simulate the effect of running a plan without actually committing it.
   * This uses shallow snapshots and heuristic deltas only.
   */
  static simulatePlan(plan: InternalPlan): CounterfactualResult {
    // Baseline snapshots
    const selBefore = SEL.getState();
    const stabBefore = StabilityMatrix.getSnapshot();
    const motBefore = MotivationEngine.compute();

    // Heuristic deltas based on goal type / intent
    let tensionDelta = 0;
    let stabilityDelta = 0;
    let urgencyDelta = 0;

    const goalType = plan.goal.type;

    if (goalType === "increase_clarity" || goalType === "consolidate_memory") {
      tensionDelta -= 0.08;
      stabilityDelta -= 0.04;
    }

    if (goalType === "run_micro_diagnostics") {
      stabilityDelta -= 0.05;
      urgencyDelta -= 0.02;
    }

    if (goalType === "stabilize_emotional_state") {
      tensionDelta -= 0.12;
      stabilityDelta -= 0.03;
    }

    if (goalType === "request_operator_input") {
      urgencyDelta -= 0.05;
    }

    if (goalType === "protect_federation_integrity") {
      stabilityDelta -= 0.1;
    }

    // Compute instability from stability score (instability = 1 - score)
    const instabilityBefore = stabBefore ? 1 - (stabBefore.score || 0.5) : 0.5;

    // Apply deltas onto baselines (for scoring only; no mutation)
    const projectedTension = Math.max(0, Math.min(1, selBefore.tension + tensionDelta));
    const projectedInstability = Math.max(
      0,
      Math.min(1, instabilityBefore + stabilityDelta)
    );
    const projectedUrgency = Math.max(
      0,
      Math.min(1, motBefore.urgency + urgencyDelta)
    );

    // Score: lower tension + lower instability + non-exploding urgency = good
    let score = 0;
    if (projectedTension < selBefore.tension) score += 0.4;
    if (projectedInstability < instabilityBefore) score += 0.4;
    if (projectedUrgency <= motBefore.urgency) score += 0.2;

    const predictedOutcome: CounterfactualResult["predictedOutcome"] =
      score > 0.6
        ? "stabilizing"
        : score < 0.3
        ? "risky"
        : "neutral";

    return {
      predictedOutcome,
      counterfactualScore: Math.max(0, Math.min(1, score)),
      deltaTension: projectedTension - selBefore.tension,
      deltaStability: projectedInstability - instabilityBefore,
      deltaMotivationUrgency: projectedUrgency - motBefore.urgency
    };
  }
}

