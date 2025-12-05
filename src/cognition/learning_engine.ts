// src/cognition/learning_engine.ts
// A73: Neural Memory Reinforcement Hooks

import { NeuralMemory } from "./neural/neural_memory_bank.ts";

export class LearningEngine {
  adjustFromReflection(reflection: any, selState: any) {
    const adjustments: any = {};

    // Get motivation from reflection if available
    const motivation = reflection.motivation || {};
    const topGoal = reflection.topGoal;

    // Example 1: If consolidation is too high too often, reduce its weight
    if (motivation.consolidation > 0.65) {
      adjustments.consolidationWeight = -0.01;
    }

    // Example 2: If clarity-seeking is rising, encourage it
    if (motivation.claritySeeking > 0.03) {
      adjustments.clarityBoost = +0.005;
    }

    // Example 3: If curiosity is high but no goal change, encourage exploration
    if (motivation.curiosity > 0.35 && topGoal?.type === "consolidate_memory") {
      adjustments.explorationBias = +0.003;
    }

    // Example 4: If tension is high, reduce it through learning
    if (selState.tension > 0.5) {
      adjustments.tensionReduction = -0.002;
    }

    // Example 5: If coherence is low, boost clarity-seeking
    if (selState.coherence < 0.4) {
      adjustments.clarityBoost = (adjustments.clarityBoost || 0) + 0.01;
    }

    if (Object.keys(adjustments).length > 0) {
      console.log("[PRIME-LEARNING] adjustments:", adjustments);
    }

    return adjustments;
  }

  // A73: Reinforcement learning from outcomes
  reinforceFromOutcome(embedding: number[], success: boolean) {
    const delta = success ? +0.1 : -0.1;
    NeuralMemory.reinforce(embedding, delta);
  }
}

