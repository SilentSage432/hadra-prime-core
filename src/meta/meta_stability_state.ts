// src/meta/meta_stability_state.ts

import type { HealingPulse } from "./healing_pulse.ts";

export class MetaStabilityState {
  lastMetaInput: string = "";
  cognitivePressure: number = 0;
  healingEvents: HealingPulse[] = [];

  applyHealing(pulse: HealingPulse) {
    if (pulse.scratchReset) this.lastMetaInput = "";
    if (pulse.reduceCognitiveLoad) this.cognitivePressure = Math.max(0, this.cognitivePressure - 1);
    this.healingEvents.push(pulse);
  }

  reset() {
    this.lastMetaInput = "";
    this.cognitivePressure = 0;
    this.healingEvents = [];
  }
}

