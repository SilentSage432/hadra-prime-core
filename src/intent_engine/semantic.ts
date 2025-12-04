import type { IntentType } from "./types.ts";
import { IntentTypes } from "./types.ts";

/**
 * Placeholder semantic scoring function.
 * Later replaced with real embeddings + vector comparisons.
 */
export function semanticScore(input: string): {
  type: IntentType;
  score: number;
} {
  const length = input.trim().length;

  // Extremely simple heuristic until embeddings arrive.
  if (length < 3) return { type: IntentTypes.UNKNOWN, score: 0.0 };
  if (length < 20) return { type: IntentTypes.DIALOG, score: 0.3 };

  return { type: IntentTypes.QUERY, score: 0.5 };
}

