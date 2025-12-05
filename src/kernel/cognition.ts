// src/kernel/cognition.ts

import { PRIMEConfig } from "../shared/config.ts";

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

