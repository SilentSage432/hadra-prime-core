// src/intent/intent_engine.ts

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
  classify(input: string): IntentResult {
    const lower = input.toLowerCase().trim();

    // Basic heuristics now — LLM-driven later
    if (/hello|hi|hey|greetings/.test(lower)) {
      return { intent: "greeting", confidence: 0.9, entities: {}, raw: input };
    }

    if (/status|running|uptime|load|how are you/.test(lower)) {
      return { intent: "ask_status", confidence: 0.9, entities: {}, raw: input };
    }

    if (/can you|please|i need you to/.test(lower)) {
      return { intent: "request_action", confidence: 0.85, entities: {}, raw: input };
    }

    if (/run|execute|perform|trigger/.test(lower)) {
      return { intent: "command", confidence: 0.8, entities: {}, raw: input };
    }

    if (/why|explain|how does/.test(lower)) {
      return { intent: "ask_explanation", confidence: 0.85, entities: {}, raw: input };
    }

    if (/i feel|i'm feeling|emotion|frustrated|excited/.test(lower)) {
      return { intent: "emotional_context", confidence: 0.75, entities: {}, raw: input };
    }

    if (/error|issue|broken|not working/.test(lower)) {
      return { intent: "report_issue", confidence: 0.85, entities: {}, raw: input };
    }

    return { intent: "unknown", confidence: 0.5, entities: {}, raw: input };
  }
}

