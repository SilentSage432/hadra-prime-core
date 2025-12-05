// A110 — Strategic Resonance Engine
// Fuses Strategy ↔ Emotion ↔ Memory ↔ Identity into unified resonance value
// This is where PRIME develops emergent preference grounded in memory, values, and identity

import type { StrategicOutcome } from "./types.ts";
import { SEL } from "../emotion/sel.ts";
import { MetaSelf } from "../cognition/self/meta_self_engine.ts";
import { Recall } from "../cognition/recall_engine.ts";
import { generateEmbedding } from "../shared/embedding.ts";

export interface ResonantOutcome extends StrategicOutcome {
  resonance: number;
  components: {
    emotional: number;
    memory: number;
    identity: number;
  };
}

export class StrategicResonanceEngine {
  private sel: typeof SEL;
  private identity: typeof MetaSelf;
  private recall: typeof Recall;

  constructor(
    sel: typeof SEL,
    identity: typeof MetaSelf,
    recall: typeof Recall
  ) {
    this.sel = sel;
    this.identity = identity;
    this.recall = recall;
  }

  // Emotional resonance - how aligned the goal feels with PRIME's emotional state
  private emotionalAlignment(goal: string): number {
    const state = this.sel.getState();
    
    // Compute alignment based on emotional coherence and valence
    // Goals that increase coherence and positive valence score higher
    const coherenceAlignment = state.coherence;
    const valenceAlignment = (state.valence + 1) / 2; // Normalize -1 to 1 -> 0 to 1
    
    // Certainty indicates confidence in the goal
    const certaintyAlignment = state.certainty;
    
    // Combined emotional alignment
    return (coherenceAlignment * 0.4 + valenceAlignment * 0.3 + certaintyAlignment * 0.3);
  }

  // Memory support - how well the goal is supported by episodic/semantic memory
  private memorySupport(goal: string): number {
    // Generate embedding for the goal to search memory
    const goalEmbedding = generateEmbedding(16);
    const memories = this.recall.recall(goalEmbedding, 10);
    
    if (!memories || memories.length === 0) return 0.1; // Small baseline for unknown goals
    
    // Average relevance of retrieved memories
    const avgRelevance = memories.reduce((sum, m) => sum + m.relevance, 0) / memories.length;
    
    // Normalize by memory count (more memories = stronger support, capped)
    const countSupport = Math.min(1, memories.length / 10);
    
    return (avgRelevance * 0.6 + countSupport * 0.4);
  }

  // Identity coherence - how well the goal fits PRIME's self-model
  private identityCoherence(goal: string): number {
    const selfModel = this.identity.exportModel();
    
    // Use stability score as base coherence indicator
    const stabilityScore = this.identity.computeStabilityScore();
    
    // Check if goal aligns with emotional profile
    const { stability, exploration, consolidation } = selfModel.emotionalProfile;
    
    // Goals that maintain stability and balance exploration/consolidation score higher
    const profileAlignment = (stability * 0.5 + (1 - Math.abs(exploration - consolidation)) * 0.5);
    
    // Combined identity coherence
    return (stabilityScore * 0.6 + profileAlignment * 0.4);
  }

  // Final resonance value computation
  computeResonance(outcome: StrategicOutcome): ResonantOutcome {
    const emotional = this.emotionalAlignment(outcome.subgoal);
    const memory = this.memorySupport(outcome.subgoal);
    const identity = this.identityCoherence(outcome.subgoal);

    const resonance =
      outcome.score * 0.4 +
      emotional * 0.2 +
      memory * 0.2 +
      identity * 0.2;

    return {
      ...outcome,
      resonance: Math.max(0, Math.min(1, resonance)), // Clamp between 0 and 1
      components: {
        emotional: Math.max(0, Math.min(1, emotional)),
        memory: Math.max(0, Math.min(1, memory)),
        identity: Math.max(0, Math.min(1, identity))
      }
    };
  }

  // Pick the best overall resonant strategy
  selectResonantStrategy(outcomes: StrategicOutcome[]): ResonantOutcome {
    if (outcomes.length === 0) {
      throw new Error("No strategic outcomes provided for resonance evaluation");
    }

    const enriched = outcomes.map(o => this.computeResonance(o));
    const sorted = enriched.sort((a, b) => b.resonance - a.resonance);
    
    return sorted[0];
  }
}

