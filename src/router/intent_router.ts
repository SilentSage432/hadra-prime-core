// src/router/intent_router.ts

import type { IntentResult } from "../intent/intent_engine.ts";

export type RouteResponse = {
  type: string;
  payload: any;
};

export class IntentRouter {
  private handlers: Map<string, Function> = new Map();

  register(intent: string, handler: Function) {
    this.handlers.set(intent, handler);
  }

  async route(intent: IntentResult): Promise<RouteResponse> {
    const handler = this.handlers.get(intent.intent);

    if (!handler) {
      return {
        type: "fallback",
        payload: { message: "No handler registered for this intent." }
      };
    }

    const result = await handler(intent);

    return { type: intent.intent, payload: result };
  }
}

