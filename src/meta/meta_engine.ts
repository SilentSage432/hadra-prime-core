// src/meta/meta_engine.ts

import type { CognitiveState } from "../cognition/cognitive_state.ts";
import type { MetaState } from "./meta_state.ts";
import { defaultMetaState } from "./meta_state.ts";
import { evaluateHeuristics } from "./heuristics.ts";

export class MetaEngine {
  evaluate(state: CognitiveState): MetaState {
    try {
      return evaluateHeuristics(state);
    } catch (err) {
      return {
        ...defaultMetaState,
        contextQuality: "low",
        actionRecommendation: "warn",
        notes: ["Meta-engine fallback activated."],
      };
    }
  }
}

export const metaEngine = new MetaEngine();

