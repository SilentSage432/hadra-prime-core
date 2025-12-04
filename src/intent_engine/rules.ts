import type { IntentType } from "./types.ts";
import { IntentTypes } from "./types.ts";

// Simple rule-based keyword mapping
const RULES: Record<string, IntentType> = {
  "status": IntentTypes.OPERATOR,
  "check": IntentTypes.OPERATOR,
  "memory": IntentTypes.OPERATOR,
  "hello": IntentTypes.DIALOG,
  "hi": IntentTypes.DIALOG,
  "who are you": IntentTypes.QUERY,
  "what": IntentTypes.QUERY,
  "deploy": IntentTypes.ACTION,
  "activate": IntentTypes.ACTION,
};

export function ruleMatch(input: string): { type: IntentType; score: number } {
  let highestScore = 0;
  let detected: IntentType = IntentTypes.UNKNOWN;

  const normalized = input.toLowerCase();

  for (const key in RULES) {
    if (normalized.includes(key)) {
      const s = key.length / normalized.length;
      if (s > highestScore) {
        highestScore = s;
        detected = RULES[key];
      }
    }
  }

  return {
    type: detected,
    score: highestScore,
  };
}

