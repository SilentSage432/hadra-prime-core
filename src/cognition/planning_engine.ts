// src/cognition/planning_engine.ts
// A74: Neural Recall Integration
// A76: Hierarchical Concept Networks

import type { ProtoGoal } from "./proto_goal_engine.ts";
import { ActionSelectionEngine } from "./action_selection_engine.ts";
import { SEL } from "../emotion/sel.ts";
import { StabilityMatrix } from "../stability/stability_matrix.ts";
import { CounterfactualEngine, type CounterfactualResult } from "../reflection/counterfactual_engine.ts";
import type { CognitiveState } from "../shared/types.ts";

export interface InternalPlanStep {
  name: string;
  fn: () => void;
}

export interface InternalPlan {
  goal: ProtoGoal;
  steps: InternalPlanStep[];
  score: number;
  counterfactual?: CounterfactualResult;
  intentModifiers?: Array<{
    type: string;
    weight: number;
    note?: string;
  }>;
}

export class PlanningEngine {
  static generatePlan(goal: ProtoGoal, cognitiveState?: CognitiveState): InternalPlan {
    // A74: Check for recall-based intuition and apply to planning
    const intentModifiers: Array<{ type: string; weight: number; note?: string }> = [];
    
    if (cognitiveState?.recall && cognitiveState.recall.intuition > 0.2) {
      intentModifiers.push({
        type: "recall_intuition",
        weight: cognitiveState.recall.intuition,
        note: "Past similar states influenced planning."
      });
      console.log("[PRIME-RECALL] Planning influenced by past experience:", {
        intuition: cognitiveState.recall.intuition.toFixed(3),
        reference: cognitiveState.recall.reference
      });
    }

    // A75: Apply concept-based bias to planning
    if (cognitiveState?.concept) {
      const conceptWeight = Math.min(0.2, cognitiveState.concept.strength * 0.05);
      intentModifiers.push({
        type: "concept_bias",
        weight: conceptWeight,
        note: `Influenced by concept: ${cognitiveState.concept.id}`
      });
      console.log("[PRIME-CONCEPT] Planning influenced by concept:", {
        conceptId: cognitiveState.concept.id,
        strength: cognitiveState.concept.strength,
        weight: conceptWeight.toFixed(3)
      });
    }

    // A76: Apply domain-based bias to planning
    if (cognitiveState?.domain) {
      const domainWeight = Math.min(0.3, cognitiveState.domain.strength * 0.05);
      intentModifiers.push({
        type: "domain_bias",
        weight: domainWeight,
        note: `Domain influence: ${cognitiveState.domain.id}`
      });
      console.log("[PRIME-DOMAIN] Planning influenced by domain:", {
        domainId: cognitiveState.domain.id,
        strength: cognitiveState.domain.strength,
        weight: domainWeight.toFixed(3)
      });
    }
    
    const action = ActionSelectionEngine.selectAction(goal);
    
    // If no action, create empty plan.
    if (!action) {
      const emptyPlan: InternalPlan = {
        goal,
        steps: [],
        score: 0
      };
      // A45: Even empty plans get counterfactual simulation
      const counterfactual = CounterfactualEngine.simulatePlan(emptyPlan);
      emptyPlan.counterfactual = counterfactual;
      return emptyPlan;
    }

    // Plans are 2–4 small steps depending on the goal type
    const steps: InternalPlanStep[] = [
      { name: "pre_stability_check", fn: () => StabilityMatrix.getSnapshot() },
      { name: goal.type, fn: action },
      { name: "post_sel_normalization", fn: () => SEL.normalize() }
    ];

    const score = this.evaluatePlan(steps, goal);

    const basePlan: InternalPlan = {
      goal,
      steps,
      score,
      intentModifiers: intentModifiers.length > 0 ? intentModifiers : undefined
    };

    // A45: Attach counterfactual simulation
    const counterfactual = CounterfactualEngine.simulatePlan(basePlan);
    basePlan.counterfactual = counterfactual;

    return basePlan;
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

