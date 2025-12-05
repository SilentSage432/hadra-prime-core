// src/cognition/planning_engine.ts

import type { ProtoGoal } from "./proto_goal_engine.ts";
import { ActionSelectionEngine } from "./action_selection_engine.ts";
import { SEL } from "../emotion/sel.ts";
import { StabilityMatrix } from "../stability/stability_matrix.ts";

export interface InternalPlanStep {
  name: string;
  fn: () => void;
}

export interface InternalPlan {
  goal: ProtoGoal;
  steps: InternalPlanStep[];
  score: number;
}

export class PlanningEngine {
  static generatePlan(goal: ProtoGoal): InternalPlan {
    const action = ActionSelectionEngine.selectAction(goal);
    
    // If no action, create empty plan.
    if (!action) {
      return {
        goal,
        steps: [],
        score: 0
      };
    }

    // Plans are 2–4 small steps depending on the goal type
    const steps: InternalPlanStep[] = [
      { name: "pre_stability_check", fn: () => StabilityMatrix.getSnapshot() },
      { name: goal.type, fn: action },
      { name: "post_sel_normalization", fn: () => SEL.normalize() }
    ];

    const score = this.evaluatePlan(steps, goal);

    return { goal, steps, score };
  }

  static evaluatePlan(steps: InternalPlanStep[], goal: ProtoGoal): number {
    let score = 0;

    // Reward plans with safety / emotional stabilization
    // Estimate impact based on goal intent (positive impact = good)
    const selState = SEL.getState();
    const goalImpact = goal.type.includes("stabilize") || goal.type.includes("consolidate") ? 1 : 0;
    score += goalImpact;

    // Reward clarity goals
    if (goal.type.includes("clarity")) score += 1;

    // Penalize if many steps (resource cost)
    if (steps.length > 3) score -= 1;

    // Reward stability-friendly plans
    const snap = StabilityMatrix.getSnapshot();
    if (snap && snap.recursionRisk === 0) score += 1;

    return score;
  }
}

