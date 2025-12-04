import type { IntentType } from "./types.ts";

export function fuseScores(
  ruleType: IntentType,
  ruleScore: number,
  semanticType: IntentType,
  semanticScore: number
): { finalType: IntentType; confidence: number } {

  const finalType =
    ruleScore >= semanticScore ? ruleType : semanticType;

  const confidence = (ruleScore * 0.6) + (semanticScore * 0.4);

  return {
    finalType,
    confidence: Math.min(1, confidence),
  };
}

