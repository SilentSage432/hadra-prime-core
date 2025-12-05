// src/safety/safety_layer.ts

import { SafetyGuard } from "./safety_guard.ts";

// Global recursion counter for safety layer
declare global {
  var recursionCounter: number;
}

if (typeof globalThis.recursionCounter === "undefined") {
  globalThis.recursionCounter = 0;
}

export function runSafetyChecks() {
  // Get recursion count from safety limiter
  const snapshot = SafetyGuard.snapshot();
  const recursionCount = snapshot.recursion || 0;
  globalThis.recursionCounter = recursionCount;

  if (recursionCount > 5) {
    return { halted: true, intensity: recursionCount };
  }

  return { halted: false };
}

