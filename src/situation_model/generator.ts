// A116 — Neural Situation Model Generator (NSMG)
// Creates dynamic, multilayered representation of PRIME's current operating context
// Unifies perception, memory, prediction, goals, desire, safety, and SEL into structured snapshot

import { PRIME_SITUATION } from "./index.ts";
import { NeuralContextEncoder } from "../neural/context_encoder.ts";
import { StabilityMatrix } from "../stability/stability_matrix.ts";
import { SEL } from "../emotion/sel.ts";
import { MemoryStore } from "../memory/memory_store.ts";
import { MemoryLayer } from "../memory/memory.ts";
import { MotivationEngine } from "../cognition/motivation_engine.ts";
import { ProtoGoalEngine } from "../cognition/proto_goal_engine.ts";

export interface SituationSnapshot {
  timestamp: number;
  micro: any;
  meso: any;
  macro: any;
  neural?: number[];
  salience: string[];
  uncertaintyScore: number;
  coherenceScore: number;
  emotionalState: any;
  recommendedFocus: string;
}

export class SituationModelGenerator {
  private encoder: NeuralContextEncoder;
  private memory: MemoryStore;

  constructor() {
    this.encoder = new NeuralContextEncoder();
    // Create memory store with a memory layer
    const memoryLayer = new MemoryLayer();
    this.memory = new MemoryStore(memoryLayer);
  }

  /**
   * Generate a complete situation snapshot combining all layers
   */
  generateSituationSnapshot(): SituationSnapshot {
    // Get current situation layers
    const micro = PRIME_SITUATION.micro.snapshot();
    const meso = PRIME_SITUATION.meso.snapshot();
    const macro = PRIME_SITUATION.macro.snapshot();

    // Combine raw situation layers
    const combined = {
      timestamp: Date.now(),
      micro,
      meso,
      macro,
    };

    // Get current cognitive state for neural encoding
    const motivation = MotivationEngine.compute();
    const goals = ProtoGoalEngine.computeGoals();
    const selState = SEL.getState();
    const stabilitySnapshot = StabilityMatrix.getSnapshot();

    // Build cognitive state for encoding
    const cognitiveState: any = {
      activeGoal: goals.length > 0 ? { type: goals[0].type } : undefined,
      confidence: selState.certainty,
      uncertainty: 1 - selState.certainty,
      motivation: motivation
    };

    // Encode into a neural representation
    const neuralContext = this.encoder.encodeContext(
      cognitiveState,
      motivation,
      stabilitySnapshot
    );

    // Determine what matters most (salience)
    const salience = this.computeSalience(neuralContext.vector, combined);

    // Compute uncertainty and coherence scores
    const uncertaintyScore = this.computeUncertainty(combined, selState, stabilitySnapshot);
    const coherenceScore = this.computeCoherence(combined, selState, stabilitySnapshot);

    // Determine recommended focus
    const recommendedFocus = this.determineRecommendedFocus(salience, combined, selState);

    return {
      ...combined,
      neural: neuralContext.vector,
      salience,
      uncertaintyScore,
      coherenceScore,
      emotionalState: selState,
      recommendedFocus
    };
  }

  /**
   * Compute salience ranking from neural encoding and situation data
   */
  private computeSalience(encoding: number[], situation: any): string[] {
    const salienceItems: Array<{ item: string; weight: number }> = [];

    // Add neural signal weights
    encoding.forEach((v, i) => {
      salienceItems.push({
        item: `neural_signal_${i}`,
        weight: Math.abs(v)
      });
    });

    // Add micro situation factors
    if (situation.micro.activeGoal) {
      salienceItems.push({
        item: `goal_${situation.micro.activeGoal}`,
        weight: situation.micro.clarity || 0.5
      });
    }

    if (situation.micro.safetyPressure > 0) {
      salienceItems.push({
        item: "safety_pressure",
        weight: situation.micro.safetyPressure
      });
    }

    if (situation.micro.cognitiveLoad > 0.7) {
      salienceItems.push({
        item: "cognitive_load",
        weight: situation.micro.cognitiveLoad
      });
    }

    // Add meso situation factors
    if (situation.meso.memoryPressure > 0.5) {
      salienceItems.push({
        item: "memory_pressure",
        weight: situation.meso.memoryPressure
      });
    }

    if (situation.meso.stabilityShift < -0.2) {
      salienceItems.push({
        item: "stability_degradation",
        weight: Math.abs(situation.meso.stabilityShift)
      });
    }

    // Add macro situation factors
    if (situation.macro.longTermObjectives.length > 0) {
      salienceItems.push({
        item: "long_term_objectives",
        weight: 0.6
      });
    }

    // Sort by weight and return top items
    return salienceItems
      .sort((a, b) => b.weight - a.weight)
      .slice(0, 10)
      .map(x => x.item);
  }

  /**
   * Compute uncertainty score from situation data
   */
  private computeUncertainty(situation: any, selState: any, stability: any): number {
    // Combine multiple uncertainty sources
    const microUncertainty = 1 - (situation.micro.clarity || 0);
    const selUncertainty = 1 - (selState.certainty || 0);
    const stabilityUncertainty = stability.score !== undefined ? 1 - stability.score : 0.5;

    // Weighted average
    return (microUncertainty * 0.4 + selUncertainty * 0.4 + stabilityUncertainty * 0.2);
  }

  /**
   * Compute coherence score from situation data
   */
  private computeCoherence(situation: any, selState: any, stability: any): number {
    // Combine multiple coherence sources
    const selCoherence = selState.coherence || 0;
    const stabilityCoherence = stability.score || 0.5;
    const mesoCoherence = situation.meso.cognitiveTrajectory === "stable" ? 0.8 : 0.5;

    // Weighted average
    return (selCoherence * 0.5 + stabilityCoherence * 0.3 + mesoCoherence * 0.2);
  }

  /**
   * Determine recommended focus based on salience and situation
   */
  private determineRecommendedFocus(
    salience: string[],
    situation: any,
    selState: any
  ): string {
    if (salience.length === 0) return "none";

    // Prioritize safety and stability
    if (salience.includes("safety_pressure")) {
      return "safety_pressure";
    }

    if (salience.includes("stability_degradation")) {
      return "stability_degradation";
    }

    if (salience.includes("memory_pressure")) {
      return "memory_pressure";
    }

    if (salience.includes("cognitive_load")) {
      return "cognitive_load";
    }

    // Check for active goals
    const goalItems = salience.filter(s => s.startsWith("goal_"));
    if (goalItems.length > 0) {
      return goalItems[0];
    }

    // Default to top salience item
    return salience[0];
  }
}

