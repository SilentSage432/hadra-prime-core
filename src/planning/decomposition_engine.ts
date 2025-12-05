// src/planning/decomposition_engine.ts

import type { PlanStep } from "./plan_engine.ts";

export type SubGoal = {
  id: string;
  name: string;
  steps: PlanStep[];
};

export class DecompositionEngine {
  /** Break a high-level goal into subgoals + steps */
  decompose(goal: string): SubGoal[] {
    console.log("[DECOMPOSE] Goal:", goal);

    // Placeholder decomposition rules — will evolve over time
    if (goal.includes("diagnose")) {
      return [
        {
          id: "subA",
          name: "Gather System Metrics",
          steps: [
            { id: "stepA1", action: "scan.metrics", status: "pending" },
            { id: "stepA2", action: "scan.memory", status: "pending" }
          ]
        },
        {
          id: "subB",
          name: "Analyze Anomalies",
          steps: [
            { id: "stepB1", action: "analyze.anomalies", status: "pending" }
          ]
        },
        {
          id: "subC",
          name: "Synthesize Report",
          steps: [
            { id: "stepC1", action: "report.system-health", status: "pending" }
          ]
        }
      ];
    }

    // Default fallback
    return [
      {
        id: "sub_default",
        name: "Default Execution Path",
        steps: [
          { id: "default_step", action: "noop", status: "pending" }
        ]
      }
    ];
  }
}

