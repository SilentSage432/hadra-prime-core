// src/meta/heuristics.ts

import type { CognitiveState } from "../cognition/cognitive_state.ts";
import type { MetaState } from "./meta_state.ts";

export function evaluateHeuristics(state: CognitiveState): MetaState {
  const notes: string[] = [];
  let certainty = 1.0;

  // --- Intent Certainty -----------------------------
  if (state.intent?.confidence !== undefined && state.intent.confidence < 0.5) {
    certainty *= state.intent.confidence;
    notes.push("Intent confidence is low.");
  }

  // --- Tone-based Uncertainty -----------------------
  if (state.tone.emotionalState === "uncertain" || state.tone.intensity < 0.3) {
    certainty *= 0.7;
    notes.push("Emotional tone suggests uncertainty or confusion.");
  }

  // --- Memory Strength ------------------------------
  // Check if memory exists and has content
  const hasMemory = state.memory && Array.isArray(state.memory) && state.memory.length > 0;
  if (!hasMemory) {
    certainty *= 0.8;
    notes.push("Relevant memory fragments are weak or incomplete.");
  }

  // --- Context Quality Rating -----------------------
  let contextQuality: MetaState["contextQuality"] = "high";
  if (certainty < 0.65) contextQuality = "medium";
  if (certainty < 0.35) contextQuality = "low";

  // --- Action Recommendation ------------------------
  let action: MetaState["actionRecommendation"] = "answer";
  if (contextQuality === "low") {
    action = "clarify";
    notes.push("Recommendation: Request clarification.");
  } else if (contextQuality === "medium") {
    action = "warn";
    notes.push("Recommendation: Answer cautiously.");
  }

  return {
    certaintyLevel: certainty,
    contextQuality,
    actionRecommendation: action,
    notes,
  };
}

