// src/cognition/attention/attention_drift_regulator.ts
// A120: Attention Drift Regulator (ADR)

import { type DriftReport } from "../../situation_model/drift_engine.ts";
import { type CognitiveState } from "../cognitive_state.ts";

export class AttentionDriftRegulator {
  private correctionWeights = {
    micro: 0.25,
    meso: 0.35,
    macro: 0.50
  };

  applyCorrection(drift: DriftReport, state: Partial<CognitiveState>) {
    if (!drift.needsCorrection || !drift.dominantDrift) return;

    const level = drift.dominantDrift;
    const weight = this.correctionWeights[level];

    console.log(`[PRIME-ADR] ${level} drift detected — applying correction weight ${weight.toFixed(3)}`);

    // Initialize fields if they don't exist
    if (state.focusBias === undefined) state.focusBias = 0;
    if (state.distractionFilterStrength === undefined) state.distractionFilterStrength = 0;
    if (state.taskCommitment === undefined) state.taskCommitment = 0;
    if (state.contextAlignment === undefined) state.contextAlignment = 0;
    if (state.missionAlignment === undefined) state.missionAlignment = 0;
    if (state.longHorizonFocus === undefined) state.longHorizonFocus = 0;

    switch (level) {
      case "micro":
        state.focusBias += weight;
        state.distractionFilterStrength += weight * 0.5;
        console.log("[PRIME-ADR] micro drift high → reinforcing task anchors.");
        break;
      case "meso":
        state.taskCommitment += weight;
        state.contextAlignment += weight * 0.3;
        console.log("[PRIME-ADR] meso drift correction applied → refocusing on active goal.");
        break;
      case "macro":
        state.missionAlignment += weight;
        state.longHorizonFocus += weight * 0.4;
        console.log("[PRIME-ADR] macro drift detected → recalibrating mission alignment weights.");
        break;
    }

    console.log("[PRIME-ADR] updated cognitive state:", {
      focusBias: state.focusBias.toFixed(3),
      taskCommitment: state.taskCommitment.toFixed(3),
      missionAlignment: state.missionAlignment.toFixed(3),
      distractionFilterStrength: state.distractionFilterStrength.toFixed(3),
      contextAlignment: state.contextAlignment.toFixed(3),
      longHorizonFocus: state.longHorizonFocus.toFixed(3)
    });
  }
}

