// src/handlers/unknown.ts

import type { IntentResult } from "../intent/intent_engine.ts";

export async function UnknownHandler(intent: IntentResult) {
  return {
    message: "I received your input, but I do not yet understand your intent."
  };
}

