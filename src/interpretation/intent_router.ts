// src/interpretation/intent_router.ts

import { ActionEngine } from "../action_layer/action_engine.ts";

export class IntentRouter {
  private actionEngine: ActionEngine;
  private kernel: any;

  constructor(actionEngine: ActionEngine, kernel?: any) {
    this.actionEngine = actionEngine;
    this.kernel = kernel;
  }

  setKernel(kernel: any) {
    this.kernel = kernel;
  }

  async route(intent: { type: string; confidence: number; payload?: any }) {
    console.log("[INTENT-ROUTER] Received:", intent);

    // A49: Handle planning intents
    if (intent.type.startsWith("action.plan.")) {
      // A50: Handle hierarchical planning
      if (intent.type === "action.plan.hierarchical") {
        if (this.kernel && this.kernel.buildAndExecuteHierarchicalPlan) {
          return this.kernel.buildAndExecuteHierarchicalPlan(intent.payload.goal);
        }
        return { routed: "error", reason: "kernel not available" };
      }
      
      // Regular planning
      if (this.kernel && this.kernel.generateAndRunPlan) {
        return this.kernel.generateAndRunPlan(
          intent.payload.goal,
          intent.payload.primaryAction,
          intent.payload.parameters
        );
      }
      return { routed: "error", reason: "kernel not available" };
    }

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

