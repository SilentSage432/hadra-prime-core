import { StabilityMatrix } from "../stability/stability_matrix.ts";
import { SafetyGuard } from "../safety/safety_guard.ts";

export function startRuntime(modules: any) {
  console.log("⏱️ HADRA-PRIME Runtime Scheduler Activated");

  setInterval(() => {
    // Heartbeat loop — future logic goes here
    
    // Pre-cognition safety check
    if (!SafetyGuard.preCognitionCheck()) {
      console.warn("[PRIME] Cognition cycle skipped due to safety constraints.");
      return; // skip cycle but keep loop alive
    }

    console.log("💓 PRIME Heartbeat");
    
    // Note: Main stability monitoring happens in PrimeEngine.tick()
    // Fusion metrics are tracked in PrimeEngine.processCommand()
    // This is a secondary monitoring point for runtime scheduler
    const snapshot = StabilityMatrix.getSnapshot();
    if (StabilityMatrix.unstable()) {
      console.warn("[PRIME] Stability degradation detected in runtime scheduler.");
    }
  }, 3000);
}

/**
 * Handle perception events with safety checks
 */
export function handlePerceptionEvent(event: any) {
  if (!SafetyGuard.prePerceptionCheck()) {
    console.warn("[PRIME] Perception event dropped for safety.");
    return;
  }

  // Process perception event
  // Future: route to perception handlers
}

