// src/cognition/cognition.ts

import { IntentEngine, type IntentResult } from "../intent/intent_engine.ts";
import { IntentRouter, type RouteResponse } from "../router/intent_router.ts";
import { SafetyGuard } from "../safety/safety_guard.ts";
import { StabilityMatrix } from "../stability/stability_matrix.ts";
import { GreetingHandler } from "../handlers/greeting.ts";
import { SystemStatusHandler } from "../handlers/system_status.ts";
import { UnknownHandler } from "../handlers/unknown.ts";

export class Cognition {
  private intentEngine = new IntentEngine();
  private router = new IntentRouter();

  constructor() {
    // Register handlers
    this.router.register("greeting", GreetingHandler);
    this.router.register("ask_status", SystemStatusHandler);
    this.router.register("unknown", UnknownHandler);
  }

  async cycle(input: string): Promise<RouteResponse> {
    // Safety check before processing
    if (!SafetyGuard.preCognitionCheck()) {
      console.warn("[PRIME-COGNITION] Safety check failed, skipping intent classification");
      return {
        type: "fallback",
        payload: { message: "Safety check failed, unable to process." }
      };
    }

    // Classify intent
    const intent = this.intentEngine.classify(input);
    console.log("[PRIME-INTENT]", intent);

    // Update stability metrics
    StabilityMatrix.update("cognition", {
      latency: 0, // Intent classification is fast
      load: intent.confidence,
      errors: intent.intent === "unknown" ? 1 : 0,
    });

    // Route to appropriate handler
    const routed = await this.router.route(intent);
    console.log("[PRIME-ROUTE]", routed);

    return routed;
  }
}

export const cognition = new Cognition();

