// src/interpretation/intent_router.ts

import { ActionEngine } from "../action_layer/action_engine.ts";

export class IntentRouter {
  private actionEngine: ActionEngine;

  constructor(actionEngine: ActionEngine) {
    this.actionEngine = actionEngine;
  }

  async route(intent: { type: string; confidence: number; payload?: any }) {
    console.log("[INTENT-ROUTER] Received:", intent);

    // non-actionable cognitive states
    if (intent.type.startsWith("thought.")) {
      return { routed: "cognitive", intent };
    }

    // actionable categories
    if (intent.type.startsWith("action.")) {
      const actionType = intent.type.replace("action.", "");
      return this.actionEngine.execute({
        type: actionType,
        payload: intent.payload,
        source: "prime",
        requiresAuth: true,
      });
    }

    return { routed: "none" };
  }
}

