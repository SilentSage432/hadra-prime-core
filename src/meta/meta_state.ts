// src/meta/meta_state.ts

export interface MetaState {
  certaintyLevel: number; // 0–1 numeric certainty
  contextQuality: "low" | "medium" | "high";
  actionRecommendation: "answer" | "clarify" | "warn" | "defer";
  notes: string[];
}

export const defaultMetaState: MetaState = {
  certaintyLevel: 1,
  contextQuality: "high",
  actionRecommendation: "answer",
  notes: [],
};

