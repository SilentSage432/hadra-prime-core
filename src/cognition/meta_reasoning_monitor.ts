// src/cognition/meta_reasoning_monitor.ts

export class MetaReasoningMonitor {
  evaluate(cycleSnapshot: any, selState: any) {
    const motivation = cycleSnapshot.motivation || {};
    
    const score = {
      coherence: 1.0,
      uncertainty: motivation.claritySeeking || 0,
      consolidationBias: motivation.consolidation || 0,
      curiosityLevel: motivation.curiosity || 0
    };

    // Grade coherence (lower consolidation = better balance)
    score.coherence = 1 - (motivation.consolidation || 0);

    const flags: any = {};

    if (score.coherence < 0.35) {
      flags.decompositionBoost = true;
    }

    if (motivation.consolidation > 0.65) {
      flags.reduceConsolidation = true;
    }

    if ((motivation.claritySeeking || 0) < 0.015) {
      flags.increaseClaritySeeking = true;
    }

    console.log("[PRIME-META] evaluation:", score, flags);

    return { score, flags };
  }
}

