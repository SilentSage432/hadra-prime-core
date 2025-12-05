// src/intent/intent_engine.ts

import { SEL } from "../emotion/sel.ts";

export type PRIMEIntent =
  | "greeting"
  | "ask_status"
  | "request_action"
  | "command"
  | "ask_explanation"
  | "emotional_context"
  | "report_issue"
  | "unknown";

export interface IntentResult {
  intent: PRIMEIntent;
  confidence: number;
  entities: Record<string, any>;
  raw: string;
}

export class IntentEngine {
  // Track long-term performance of each intent
  private static intentStats: Record<
    string,
    { success: number; failure: number; confusion: number }
  > = {};

  // Gravitation weights — how strongly PRIME "leans" toward successful intents
  private static gravitationWeights = {
    successPull: 0.05,
    failurePush: 0.04,
    confusionDampen: 0.02,
  };

  // Returns a gravitation score for an intent
  private static getIntentGravity(intent: string): number {
    const stats = this.intentStats[intent] || {
      success: 0,
      failure: 0,
      confusion: 0,
    };
    const total = stats.success + stats.failure + stats.confusion || 1;
    const successRatio = stats.success / total;
    const failureRatio = stats.failure / total;
    const confusionRatio = stats.confusion / total;

    // Gravitation score: higher = PRIME is biased toward this intent
    return (
      successRatio * this.gravitationWeights.successPull -
      failureRatio * this.gravitationWeights.failurePush -
      confusionRatio * this.gravitationWeights.confusionDampen
    );
  }

  // Update intent performance stats
  static reinforceIntent(intent: string, outcome: "success" | "failure" | "confusion") {
    if (!this.intentStats[intent]) {
      this.intentStats[intent] = { success: 0, failure: 0, confusion: 0 };
    }
    this.intentStats[intent][outcome]++;
  }

  classify(input: string): IntentResult {
    const lower = input.toLowerCase().trim();

    // Basic heuristics now — LLM-driven later
    let intentResult: IntentResult;
    
    if (/hello|hi|hey|greetings/.test(lower)) {
      intentResult = { intent: "greeting", confidence: 0.9, entities: {}, raw: input };
    } else if (/status|running|uptime|load|how are you/.test(lower)) {
      intentResult = { intent: "ask_status", confidence: 0.9, entities: {}, raw: input };
    } else if (/can you|please|i need you to/.test(lower)) {
      intentResult = { intent: "request_action", confidence: 0.85, entities: {}, raw: input };
    } else if (/run|execute|perform|trigger/.test(lower)) {
      intentResult = { intent: "command", confidence: 0.8, entities: {}, raw: input };
    } else if (/why|explain|how does/.test(lower)) {
      intentResult = { intent: "ask_explanation", confidence: 0.85, entities: {}, raw: input };
    } else if (/i feel|i'm feeling|emotion|frustrated|excited/.test(lower)) {
      intentResult = { intent: "emotional_context", confidence: 0.75, entities: {}, raw: input };
    } else if (/error|issue|broken|not working/.test(lower)) {
      intentResult = { intent: "report_issue", confidence: 0.85, entities: {}, raw: input };
    } else {
      intentResult = { intent: "unknown", confidence: 0.5, entities: {}, raw: input };
    }
    
    // Apply gravitation scoring
    const gravity = this.getIntentGravity(intentResult.intent);
    (intentResult as any).gravity = gravity;
    
    return intentResult;
  }

  computeIntentScore(intent: IntentResult): number {
    const emotion = SEL.getState();

    // Base score from intent fields
    let score = intent.confidence ?? 0.5;

    // Emotion influence:
    // -------------------------------------
    // Higher certainty → higher clarity
    score += (emotion.certainty - 0.5) * 0.4;

    // Higher valence → more positive alignment
    score += emotion.valence * 0.3;

    // Higher tension → reduce intent confidence
    score -= emotion.tension * 0.5;

    // Coherence improves stability of scoring
    score += (emotion.coherence - 0.5) * 0.2;

    // Clamp between 0 and 1
    return Math.max(0, Math.min(1, score));
  }
}

