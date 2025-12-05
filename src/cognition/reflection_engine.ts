// src/cognition/reflection_engine.ts

export class ReflectionEngine {
  reflect(cognitiveState: any, selState: any) {
    const reflection = {
      timestamp: Date.now(),
      motivation: cognitiveState.motivation,
      topGoal: cognitiveState.topGoal,
      sel: selState,
      summary: this.generateSummary(cognitiveState, selState)
    };

    console.log("[PRIME-REFLECTION]", reflection.summary);

    return reflection;
  }

  private generateSummary(cog: any, sel: any): string {
    const goal = cog.topGoal?.type || "none";
    const motivation = cog.motivation || {};
    
    return `Reflecting: PRIME prioritized '${goal}' due to ` +
           `consolidation=${(motivation.consolidation || 0).toFixed(3)}, ` +
           `curiosity=${(motivation.curiosity || 0).toFixed(3)}, ` +
           `claritySeeking=${(motivation.claritySeeking || 0).toFixed(3)}.`;
  }
}

