// src/reflection/self_consistency.ts

export class SelfConsistencyEngine {
  review(plan: any) {
    const issues: string[] = [];

    // ---------------------------------------------------
    // 1. Coherence Check
    // ---------------------------------------------------
    if (!plan.strategicGoal || !plan.operationalPlan) {
      issues.push("Plan structure incomplete.");
    }

    if (plan.operationalPlan?.workflow?.length === 0) {
      issues.push("Operational plan lacks actionable structure.");
    }

    // Simple mismatch detection
    if (plan.intent && !plan.strategicGoal.highLevelObjective.includes(plan.intent)) {
      issues.push("Strategic intent mismatch.");
    }

    // ---------------------------------------------------
    // 2. Completeness Check
    // ---------------------------------------------------
    if (!plan.tacticalSteps || plan.tacticalSteps.length < 2) {
      issues.push("Insufficient tactical steps.");
    }

    if (!plan.atomicActions || plan.atomicActions.length === 0) {
      issues.push("No atomic actions available.");
    }

    // ---------------------------------------------------
    // 3. Alignment Check (Operator Intent)
    // ---------------------------------------------------
    const alignmentScore = this.estimateAlignment(plan.intent, plan.strategicGoal);

    return {
      issues,
      alignmentScore,
      isConsistent: issues.length === 0 && alignmentScore > 0.75,
      requiresOperatorReview: issues.length > 0 || alignmentScore < 0.75
    };
  }

  estimateAlignment(intent: string, goal: any) {
    if (!goal?.highLevelObjective) return 0.0;

    return intent === goal.highLevelObjective ? 0.95 : 0.65;
  }
}

