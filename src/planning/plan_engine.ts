// src/planning/plan_engine.ts

import { DecompositionEngine, type SubGoal } from "./decomposition_engine.ts";

export type PlanStep = {
  id: string;
  action: string;
  params?: any;
  status: "pending" | "running" | "success" | "failed";
};

export type Plan = {
  id: string;
  goal: string;
  steps: PlanStep[];
  currentStep: number;
  status: "pending" | "running" | "complete" | "aborted";
};

export class PlanEngine {
  private plans: Map<string, Plan> = new Map();
  private decomposition = new DecompositionEngine();

  /** Create a new plan object from a high-level operator intent */
  createPlan(goal: string, steps: PlanStep[]): Plan {
    const plan: Plan = {
      id: `plan_${Date.now()}`,
      goal,
      steps,
      currentStep: 0,
      status: "pending",
    };

    this.plans.set(plan.id, plan);
    console.log("[PLAN] Created:", plan);

    return plan;
  }

  /** Begin executing a plan */
  async executePlan(plan: Plan, actionEngine: any) {
    console.log(`[PLAN] Starting: ${plan.goal}`);
    plan.status = "running";

    while (plan.currentStep < plan.steps.length) {
      const step = plan.steps[plan.currentStep];
      step.status = "running";

      try {
        console.log(`[PLAN] Executing step ${step.id}: ${step.action}`);
        await actionEngine.execute({
          type: step.action,
          payload: step.params,
          source: "prime",
          requiresAuth: false,
        });
        step.status = "success";
      } catch (err) {
        console.log(`[PLAN] Step failed: ${step.id}`, err);
        step.status = "failed";
        plan.status = "aborted";
        return;
      }

      plan.currentStep++;
    }

    plan.status = "complete";
    console.log("[PLAN] Completed:", plan.id);
  }

  /** Get a plan by ID */
  getPlan(id: string): Plan | undefined {
    return this.plans.get(id);
  }

  /** Get all active plans */
  getActivePlans(): Plan[] {
    return Array.from(this.plans.values()).filter(
      (p) => p.status === "running" || p.status === "pending"
    );
  }

  /** A50: Create hierarchical plan from goal using decomposition engine */
  public createHierarchicalPlan(goal: string): Plan {
    const subGoals = this.decomposition.decompose(goal);
    const allSteps = subGoals.flatMap(sg => sg.steps);

    const plan: Plan = {
      id: `plan_${Date.now()}`,
      goal,
      steps: allSteps,
      currentStep: 0,
      status: "pending"
    };

    this.plans.set(plan.id, plan);
    console.log("[PLAN] Hierarchical plan created:", plan);
    return plan;
  }
}

