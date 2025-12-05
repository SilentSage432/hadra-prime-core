// src/motivation/long_horizon_engine.ts

export class LongHorizonIntentEngine {
  decayIntent(intent: any) {
    const decayRate = intent.decayRate ?? 0.04;
    const reinforcement = intent.reinforcement ?? 0;
    intent.strength = intent.strength * (1 - decayRate) + reinforcement;
  }

  isExpired(intent: any) {
    const expired =
      intent.strength < 0.01 ||
      (intent.horizon && Date.now() > intent.createdAt + intent.horizon);

    return expired;
  }

  stabilize(smv: any) {
    if (!smv.longHorizonIntentions) {
      smv.longHorizonIntentions = [];
    }

    // decay + reinforce
    smv.longHorizonIntentions.forEach((intent: any) => {
      this.decayIntent(intent);
    });

    // remove expired
    const beforeCount = smv.longHorizonIntentions.length;
    smv.longHorizonIntentions = smv.longHorizonIntentions.filter(
      (intent: any) => !this.isExpired(intent)
    );
    
    const expiredCount = beforeCount - smv.longHorizonIntentions.length;
    if (expiredCount > 0) {
      console.log(`[LHIS] ${expiredCount} long-horizon intention(s) expired.`);
    }
  }

  integrateNewIntentions(smv: any) {
    if (!smv.activeIntentions || smv.activeIntentions.length === 0) return;

    if (!smv.longHorizonIntentions) {
      smv.longHorizonIntentions = [];
    }

    smv.activeIntentions.forEach((intent: any) => {
      // Check if similar intention already exists
      const existing = smv.longHorizonIntentions.find(
        (i: any) => i.type === intent.type
      );

      if (existing) {
        // Reinforce existing intention
        existing.reinforcement = (existing.reinforcement || 0) + 0.02;
        existing.strength = Math.min(1.0, existing.strength + 0.01);
      } else {
        // Add new intention
        smv.longHorizonIntentions.push({
          ...intent,
          createdAt: Date.now(),
          decayRate: 0.03,
          reinforcement: 0.02
        });
      }
    });
  }
}

