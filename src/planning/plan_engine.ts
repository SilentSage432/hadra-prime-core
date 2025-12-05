// src/planning/plan_engine.ts
// A124: Extended with uncertainty-aware planning

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

  /** A124: Generate plan with uncertainty awareness */
  public generatePlan(goal: any, state: any): any {
    // A124: Check uncertainty and generate fallback plan if needed
    if (state?.uncertaintyScore !== undefined && state.uncertaintyScore > 0.6) {
      console.log("[PRIME-PLANNING] High uncertainty → generating safer fallback plan.");
      return this.generateFallbackPlan(goal);
    }
    
    if (state?.uncertaintyScore !== undefined && state.uncertaintyScore > 0.4) {
      console.log("[PRIME-PLANNING] Uncertainty moderate → keeping actions conservative.");
    }

    // Default plan generation (placeholder - actual implementation would be here)
    return {
      goal: goal,
      steps: [],
      score: 0.7,
      counterfactual: null
    };
  }

  /** A124: Generate a safer fallback plan when uncertainty is high */
  private generateFallbackPlan(goal: any): any {
    return {
      goal: goal,
      steps: [
        { id: "step1", name: "Gather more information", action: "seek_clarity", params: {} },
        { id: "step2", name: "Request operator guidance", action: "request_guidance", params: {} }
      ],
      score: 0.5,
      counterfactual: "High uncertainty requires conservative approach",
      isFallback: true
    };
  }
}

