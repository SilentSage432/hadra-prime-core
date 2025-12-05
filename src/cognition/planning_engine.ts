// src/cognition/planning_engine.ts
// A74: Neural Recall Integration
// A76: Hierarchical Concept Networks
// A77: Knowledge Graph Integration
// A78: Inference Engine Integration
// A79: Long-Range Predictive Model

import type { ProtoGoal } from "./proto_goal_engine.ts";
import { Knowledge } from "./knowledge/knowledge_graph.ts";
import { Inference } from "./inference/inference_engine.ts";
import { Foresight } from "./prediction/foresight_engine.ts";
import { SEL } from "../emotion/sel.ts";
import { ActionSelectionEngine } from "./action_selection_engine.ts";
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

    // A78: Use inference engine for strategy recommendations
    if (cognitiveState?.embedding) {
      const inferred = Inference.inferStrategy(cognitiveState.embedding);
      if (inferred && inferred.confidence > 0.3) {
        intentModifiers.push({
          type: "inference_strategy",
          weight: inferred.confidence,
          note: inferred.strategy,
        });
        console.log("[PRIME-INFERENCE] Planning influenced by inferred strategy:", {
          strategy: inferred.strategy,
          confidence: inferred.confidence.toFixed(3),
          weight: inferred.confidence.toFixed(3)
        });
      }
    }

    // A79: Use foresight predictions to influence planning
    if (cognitiveState?.motivation) {
      const selState = SEL.getState();
      try {
        const fs = Foresight.forecastSystemState(cognitiveState.motivation, selState);

        // If future stability pressure is rising, bias toward stabilization plans
        if (fs.projectedSEL.length > 10 && fs.projectedSEL[10].stabilityPressure && fs.projectedSEL[10].stabilityPressure > 0.05) {
          intentModifiers.push({
            type: "foresight_stabilization_bias",
            weight: 0.15,
            note: "Upcoming stability pressure detected",
          });
          console.log("[PRIME-FORESIGHT] Planning biased toward stabilization:", {
            projectedPressure: fs.projectedSEL[10].stabilityPressure.toFixed(4)
          });
        }

        // If future curiosity is rising, bias towards exploration
        if (fs.projectedMotivation.length > 10) {
          const futureCuriosity = fs.projectedMotivation[10].curiosity;
          const currentCuriosity = cognitiveState.motivation.curiosity;
          
          if (futureCuriosity > currentCuriosity) {
            intentModifiers.push({
              type: "foresight_exploration_bias",
              weight: 0.1,
              note: "Long-term curiosity growth predicted",
            });
            console.log("[PRIME-FORESIGHT] Planning biased toward exploration:", {
              currentCuriosity: currentCuriosity.toFixed(3),
              projectedCuriosity: futureCuriosity.toFixed(3)
            });
          }
        }

        // If clarity-seeking is projected to rise significantly, bias toward clarity plans
        if (fs.projectedMotivation.length > 10) {
          const futureClarity = fs.projectedMotivation[10].claritySeeking;
          const currentClarity = cognitiveState.motivation.claritySeeking;
          
          if (futureClarity > currentClarity + 0.1) {
            intentModifiers.push({
              type: "foresight_clarity_bias",
              weight: 0.12,
              note: "Significant clarity-seeking growth predicted",
            });
          }
        }
      } catch (error) {
        console.warn("[PRIME-FORESIGHT] Error during planning prediction:", error);
      }
    }

    // A77: Reference knowledge graph for planning influence
    const graphStats = Knowledge.getStats();
    if (graphStats.nodeCount > 0) {
      // Lightweight graph reference for planning (stats only, not full graph)
      (cognitiveState as any).knowledgeInfluence = {
        nodeCount: graphStats.nodeCount,
        edgeCount: graphStats.edgeCount,
        nodesByType: graphStats.nodesByType
      };
      console.log("[PRIME-KNOWLEDGE] Planning with knowledge graph context:", {
        nodes: graphStats.nodeCount,
        edges: graphStats.edgeCount
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

