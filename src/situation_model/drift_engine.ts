// src/situation_model/drift_engine.ts
// A119: Contextual Attachment & Drift Detection Across Timescales

import { MicroSituationModel } from "./micro.ts";
import { MesoSituationModel } from "./meso.ts";
import { MacroSituationModel } from "./macro.ts";

// Define state type here to avoid circular dependency
interface TimescaleSituationState {
  micro: MicroSituationModel | null;
  meso: MesoSituationModel | null;
  macro: MacroSituationModel | null;
}

export interface DriftReport {
  microDrift: number;
  mesoDrift: number;
  macroDrift: number;
  needsCorrection: boolean;
  dominantDrift: "micro" | "meso" | "macro" | null;
}

export class DriftEngine {
  private thresholds = {
    micro: 0.35,
    meso: 0.30,
    macro: 0.25
  };

  computeDrift(state: TimescaleSituationState): DriftReport {
    const microDrift = (state.micro as any)?.drift?.() ?? 0;
    const mesoDrift = (state.meso as any)?.drift?.() ?? 0;
    const macroDrift = (state.macro as any)?.drift?.() ?? 0;

    const needsCorrection =
      microDrift > this.thresholds.micro ||
      mesoDrift > this.thresholds.meso ||
      macroDrift > this.thresholds.macro;

    let dominant: DriftReport["dominantDrift"] = null;
    const max = Math.max(microDrift, mesoDrift, macroDrift);

    if (max === microDrift && max > 0) dominant = "micro";
    else if (max === mesoDrift && max > 0) dominant = "meso";
    else if (max === macroDrift && max > 0) dominant = "macro";

    return {
      microDrift,
      mesoDrift,
      macroDrift,
      needsCorrection,
      dominantDrift: dominant
    };
  }
}

