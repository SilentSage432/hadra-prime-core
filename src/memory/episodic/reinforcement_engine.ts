// src/memory/episodic/reinforcement_engine.ts
// A68: Episodic Reinforcement Engine

import type { Episode } from "./episode_builder.ts";

export interface ReinforcementSignal {
  clarityDelta: number;
  consolidationDelta: number;
  stabilityDelta: number;
  predictionSuppressionScore: number;
}

export class EpisodicReinforcementEngine {
  computeReinforcement(episode: Episode): ReinforcementSignal {
    if (!episode.events.length) {
      return {
        clarityDelta: 0,
        consolidationDelta: 0,
        stabilityDelta: 0,
        predictionSuppressionScore: 1
      };
    }

    const first = episode.events[0];
    const last = episode.events[episode.events.length - 1];

    // Extract clarity from reflection (using coherence as proxy if clarity not available)
    const firstClarity = (first.reflection as any)?.clarity ?? (first.reflection as any)?.sel?.coherence ?? 0;
    const lastClarity = (last.reflection as any)?.clarity ?? (last.reflection as any)?.sel?.coherence ?? 0;
    const clarityDelta = lastClarity - firstClarity;

    // Extract consolidation from motivation
    const firstConsolidation = (first.motivation as any)?.consolidation ?? 0;
    const lastConsolidation = (last.motivation as any)?.consolidation ?? 0;
    const consolidationDelta = lastConsolidation - firstConsolidation;

    // Extract stability from stability snapshot
    const firstStability = (first.stability as any)?.score ?? (first.stability as any)?.stabilityScore ?? 0;
    const lastStability = (last.stability as any)?.score ?? (last.stability as any)?.stabilityScore ?? 0;
    const stabilityDelta = lastStability - firstStability;

    // Prediction suppression based on episode length
    const predictionSuppressionScore = Math.max(
      0,
      1 - Math.abs(episode.events.length / 1000)
    );

    return {
      clarityDelta,
      consolidationDelta,
      stabilityDelta,
      predictionSuppressionScore
    };
  }
}

