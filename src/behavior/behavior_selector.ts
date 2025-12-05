// src/behavior/behavior_selector.ts
// A124: Extended with uncertainty-aware behavior selection

export class BehaviorSelector {
  selectBehavior(smv: any, cognitiveState?: any) {
    // A124: Check uncertainty and override behavior if needed
    if (cognitiveState?.uncertaintyScore !== undefined) {
      if (cognitiveState.uncertaintyScore > 0.75) {
        console.log("[PRIME-BEHAVIOR] High uncertainty detected → shifting to 'seek_clarity' mode.");
        return { type: "seek_clarity", reason: "High uncertainty detected" };
      }
      if (cognitiveState.uncertaintyScore > 0.55) {
        console.log("[PRIME-BEHAVIOR] Moderate uncertainty detected → shifting to 'pause_and_reflect' mode.");
        return { type: "pause_and_reflect", reason: "Moderate uncertainty" };
      }
    }
    const desire = smv.desireState || { recentValence: 0, cumulativeValence: 0 };

    const rv = desire.recentValence;
    const cv = desire.cumulativeValence;

    // behavior weights influenced by proto-desire
    const weights: any = {
      consolidate_memory: 0.4 + (-rv * 0.3) + (-cv * 0.2),
      seek_clarity:       0.3 + (rv * 0.4),
      explore:            0.2 + (rv * 0.6) + (cv * 0.3),
      stabilize:          0.6 + (-rv * 0.2) + (-cv * 0.3),
      reduce_drift:       0.7 + (-rv * 0.4),
      reflect:            0.5 + (cv * 0.5)
    };

    // Ensure all weights are non-negative
    Object.keys(weights).forEach(key => {
      weights[key] = Math.max(0, weights[key]);
    });

    // normalize
    const total = Object.values(weights).reduce((a: number, b: number) => a + b, 0);
    const normalized = Object.entries(weights).map(([k, v]) => [k, (v as number) / total]);

    // choose based on highest weight
    const behavior = normalized.sort((a: any, b: any) => b[1] - a[1])[0][0];

    return behavior;
  }
}

