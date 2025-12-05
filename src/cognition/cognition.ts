// src/cognition/cognition.ts

import { IntentEngine, type IntentResult } from "../intent/intent_engine.ts";
import { IntentRouter, type RouteResponse } from "../router/intent_router.ts";
import { SafetyGuard } from "../safety/safety_guard.ts";
import { StabilityMatrix } from "../stability/stability_matrix.ts";
import { SEL } from "../emotion/sel.ts";
import { MotivationEngine } from "./motivation_engine.ts";
import { ProtoGoalEngine } from "./proto_goal_engine.ts";
import { ActionSelectionEngine } from "./action_selection_engine.ts";
import { PlanningEngine } from "./planning_engine.ts";
import { GreetingHandler } from "../handlers/greeting.ts";
import { SystemStatusHandler } from "../handlers/system_status.ts";
import { UnknownHandler } from "../handlers/unknown.ts";
import { Recall } from "./recall_engine.ts";
import type { CognitiveState } from "../shared/types.ts";
import { NeuralContextEncoder } from "../neural/context_encoder.ts";

export class Cognition {
  private intentEngine = new IntentEngine();
  private router = new IntentRouter();

  constructor() {
    // Register handlers
    this.router.register("greeting", GreetingHandler);
    this.router.register("ask_status", SystemStatusHandler);
    this.router.register("unknown", UnknownHandler);
  }

  async cycle(input: string): Promise<RouteResponse> {
    // Safety check before processing
    if (!SafetyGuard.preCognitionCheck()) {
      console.warn("[PRIME-COGNITION] Safety check failed, skipping intent classification");
      return {
        type: "fallback",
        payload: { message: "Safety check failed, unable to process." }
      };
    }

    // Classify intent
    const intent = this.intentEngine.classify(input);
    console.log("[PRIME-INTENT]", intent);

    // A39: Track last intent for motivation computation
    MotivationEngine.setLastIntent(intent);

    // A38: Apply SEL influence from gravitation
    const gravity = (intent as any).gravity || 0;
    SEL.applyIntentInfluence(gravity);

    // A39: Compute motivation vector each cognition cycle
    const motivation = MotivationEngine.compute();
    console.log("[PRIME-MOTIVATION]", motivation);

    // A40: Compute proto-goals
    const goals = ProtoGoalEngine.computeGoals();
    console.log("[PRIME-GOALS]", goals);

    // A74/A79: Create cognitive state and perform recall before planning
    const topGoal = goals[0] || null;
    const cognitiveState: any = {
      activeGoal: topGoal ? { type: topGoal.type } : undefined,
      confidence: intent.confidence,
      uncertainty: 1 - intent.confidence,
      motivation: motivation // A79: Include motivation for foresight predictions
    };

    // A74: Perform preliminary recall lookup for planning
    if (topGoal) {
      // Build a simple embedding for recall lookup
      const ncel = new NeuralContextEncoder();
      const baseFeatures = [
        motivation.urgency,
        motivation.curiosity,
        motivation.claritySeeking,
        motivation.consolidation,
        intent.confidence,
        1 - intent.confidence
      ];
      // Create a simple lookup vector (pad to 64 for similarity search)
      const lookupVector = new Array(64).fill(0);
      baseFeatures.forEach((v, i) => {
        if (i < lookupVector.length) {
          lookupVector[i] = Math.max(-1, Math.min(1, v));
        }
      });

      const recallResults = Recall.recall(lookupVector, 3);
      const recallSummary = Recall.summarizeRecall(recallResults);
      if (recallSummary) {
        cognitiveState.recall = recallSummary;
      }
    }

    // A42/A75: Generate plans for each goal (with recall-informed and concept-informed cognitive state)
    // A75: Match concept before planning if embedding is available
    if (cognitiveState.recall && cognitiveState.recall.reference) {
      // If we have recall, we likely have an embedding in the neural context
      // For now, concepts will be matched during reflection, but we could do it here too
    }
    
    const plans = goals.map((g) => PlanningEngine.generatePlan(g, cognitiveState));
    console.log("[PRIME-PLANS]", plans);

    // A42: Choose best plan by score
    const best = plans.sort((a, b) => b.score - a.score)[0] || null;
    if (best && best.steps.length > 0) {
      console.log("[PRIME-PLAN] Selected plan:", {
        goal: best.goal.type,
        score: best.score,
        steps: best.steps.map((s) => s.name),
        counterfactual: best.counterfactual
      });
      // Execute plan steps one-by-one
      for (const step of best.steps) {
        console.log("[PRIME-PLAN-STEP] Executing:", step.name);
        try {
          step.fn();
        } catch (err) {
          console.error("[PRIME-PLAN-ERROR]", err);
          break;
        }
      }
    }

    // Update stability metrics
    StabilityMatrix.update("cognition", {
      latency: 0, // Intent classification is fast
      load: intent.confidence,
      errors: intent.intent === "unknown" ? 1 : 0,
    });

    // Route to appropriate handler
    const routed = await this.router.route(intent);
    console.log("[PRIME-ROUTE]", routed);

    // A35: Post-cognition normalization
    this.finalizeCognition(routed, intent);

    return routed;
  }

  finalizeCognition(result: RouteResponse, intent: IntentResult) {
    const emotion = SEL.getState();
    
    // A35: cognition improves coherence gradually
    const newCoherence = Math.min(1, emotion.coherence + 0.02);
    
    // Reduce tension after successful cognition pass
    const newTension = Math.max(0, emotion.tension - 0.03);
    
    // Update emotion state
    SEL.updateEmotion({
      stability: newCoherence,
      tensionSignal: newTension,
    });

    // A36: Determine cognition outcome
    let outcome: "success" | "confusion" | "failure" = "success";
    if (result && (result as any).qualityScore !== undefined) {
      const qualityScore = (result as any).qualityScore;
      if (qualityScore > 0.8) outcome = "success";
      else if (qualityScore > 0.4) outcome = "confusion";
      else outcome = "failure";
    } else {
      // Heuristic: if result type is "fallback" or "unknown", treat as confusion/failure
      if (result.type === "fallback") {
        outcome = "failure";
      } else if (result.type === "unknown") {
        outcome = "confusion";
      } else {
        // Default to success for known intent types
        outcome = "success";
      }
    }

    // Apply emotional reinforcement
    SEL.reinforce(outcome);

    // A38: Reinforce intent gravitation based on outcome
    IntentEngine.reinforceIntent(intent.intent, outcome);

    // Log outcome for debugging
    console.log(`[PRIME-SEL] Reinforcement outcome: ${outcome}`);
  }

  selectReasoningPath(paths: Array<{ type: string; weight: number; [key: string]: any }>) {
    const emotion = SEL.getState();

    return paths
      .map((p) => {
        let modified = p.weight;

        // Emotion-aware modulation:
        modified += (emotion.valence - 0.5) * 0.2;
        modified += (emotion.certainty - 0.5) * 0.3;

        // High tension pushes PRIME toward cautionary paths
        if (emotion.tension > 0.6 && p.type === "cautious") {
          modified += 0.3;
        }

        // Low coherence penalizes high-risk paths
        if (emotion.coherence < 0.4 && p.type === "aggressive") {
          modified -= 0.3;
        }

        return { ...p, adjustedWeight: modified };
      })
      .sort((a, b) => b.adjustedWeight - a.adjustedWeight)[0];
  }
}

export const cognition = new Cognition();

