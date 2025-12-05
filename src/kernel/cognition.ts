// src/kernel/cognition.ts

import { PRIMEConfig } from "../shared/config.ts";
import { MultiTimescaleSituationModel } from "../situation_model/multi_timescale_model.ts";
import { PRIME_SITUATION } from "../situation_model/index.ts";

let lastCycle = Date.now();
let cooldownActive = false;

export function shouldProcessCognition(): boolean {
  const now = Date.now();

  // 1. If cooldown is active → suppress cognition
  if (cooldownActive) {
    if (now - lastCycle >= PRIMEConfig.cognition.cooldownMs) {
      cooldownActive = false; // cooldown expired
    } else {
      return false; // still cooling
    }
  }

  // 2. Suppress if there is no stimulus
  if (!PRIMEConfig.runtime.hasStimulus) {
    return false;
  }

  lastCycle = now;
  return true;
}

export function triggerCooldown(intensity = 1) {
  cooldownActive = true;
  // Expand cooldown duration based on recursion severity
  PRIMEConfig.cognition.cooldownMs =
    PRIMEConfig.cognition.baseCooldownMs * intensity;
}

// A118: PrimeCognition class for multi-timescale situation modeling
export class PrimeCognition {
  private timescaleModel = new MultiTimescaleSituationModel();

  constructor() {
    // Initialize with existing situation models from PRIME_SITUATION
    this.timescaleModel.updateMicro(PRIME_SITUATION.micro);
    this.timescaleModel.updateMeso(PRIME_SITUATION.meso);
    this.timescaleModel.updateMacro(PRIME_SITUATION.macro);
  }

  updateSituations(micro?: any, meso?: any, macro?: any) {
    // Update timescale model with current situation model instances
    // PRIME_SITUATION already maintains the instances, we just sync references
    if (micro) this.timescaleModel.updateMicro(micro);
    if (meso) this.timescaleModel.updateMeso(meso);
    if (macro) this.timescaleModel.updateMacro(macro);
  }

  getSituationState() {
    return this.timescaleModel.getState();
  }

  getSituationSummary() {
    return this.timescaleModel.summarize();
  }

  // A119: Check for drift across all timescales
  driftCheck() {
    const report = this.timescaleModel.analyzeDrift();
    if (report.needsCorrection) {
      console.log("[PRIME-DRIFT] Drift detected:", {
        micro: report.microDrift.toFixed(3),
        meso: report.mesoDrift.toFixed(3),
        macro: report.macroDrift.toFixed(3),
        dominant: report.dominantDrift
      });
      // corrective behavior: re-anchor cognitive focus
      this.realignToSituation(report.dominantDrift);
    }
  }

  // A119: Realign cognitive focus to the specified timescale layer
  realignToSituation(layer: "micro" | "meso" | "macro" | null) {
    if (!layer) return;
    
    console.log(`[PRIME-DRIFT] realigning focus to ${layer} layer...`);
    
    // Future: Implement actual realignment logic
    // For now, this logs the realignment intent
    switch (layer) {
      case "micro":
        console.log("[PRIME-DRIFT] micro drift high: refocusing...");
        break;
      case "meso":
        console.log("[PRIME-DRIFT] meso drift rising: reconnecting to task context.");
        break;
      case "macro":
        console.log("[PRIME-DRIFT] macro drift detected: recalibrating mission alignment.");
        break;
    }
  }
}

