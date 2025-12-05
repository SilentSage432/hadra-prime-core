// HADRA-PRIME Kernel Entry Point
// This file simply boots PRIME Core and exposes no internal modules.

import PRIME from "../prime.ts"; // triggers PRIME initialization + log emit
import { StabilityMatrix } from "../stability/stability_matrix.ts";
import { ThreadPool } from "./threads/thread_pool.ts";
import { anchorRegistry, resonanceBus } from "../memory/index.ts";
import { generateEmbedding } from "../shared/embedding.ts";
import { harmonizationBus, harmonizationEngine, type IntentSignature } from "../intent_engine/harmonization.ts";
import { phaseScheduler, type CognitivePhase } from "./phase_scheduler.ts";
import { SafetyGuard } from "../safety/safety_guard.ts";
import { PredictiveCoherence } from "../cognition/predictive_coherence.ts";
import { reactiveLattice } from "./reactive_lattice.ts";
import { PRIME_LOOP_GUARD } from "../shared/loop_guard.ts";
import { safetyEngine } from "../safety/safety_engine.ts";
import { predictEngine } from "../prediction/predict_engine.ts";
import { phaseEngine } from "../phase/phase_engine.ts";
import { shouldProcessCognition, triggerCooldown } from "./cognition.ts";
import { runSafetyChecks } from "../safety/safety_layer.ts";
import { runPrediction } from "../prediction/predict.ts";
import { runInterpretation } from "../interpretation/dispatcher.ts";
import { PRIMEConfig } from "../shared/config.ts";
import { cognitiveLoop } from "./cognitive_loop.ts";
import crypto from "crypto";

console.log("[PRIME] Initializing Stability Matrix...");
StabilityMatrix.init();

console.log("[PRIME] Cognitive Fusion Stability Layer active.");

console.log("[PRIME] Initializing cognitive threads...");
// ThreadPool will be initialized with default instances
// For full integration, it should be initialized with PRIME's actual instances
ThreadPool.init();

console.log("[KERNEL] HADRA-PRIME core boot sequence complete.");

// Start event-driven cognitive loop
cognitiveLoop.start();

// Intent harmonization broadcasting function
function broadcastIntent(intent: any) {
  const sig: IntentSignature = {
    id: crypto.randomUUID(),
    timestamp: Date.now(),
    category: intent.type || "unknown",
    purpose: intent.type || "general",
    confidence: intent.confidence || 0.5,
    vector: generateEmbedding(16),
    weight: Math.random() * 0.4
  };

  harmonizationBus.publish(sig);
  harmonizationEngine.ingest(sig);
  const harmonized = harmonizationEngine.harmonize(sig);
  console.log("[HARMONIZATION] alignment:", harmonized.alignment.toFixed(3),
              " adjusted:", harmonized.adjustedConfidence.toFixed(3));
}

// Export for use in cognitive threads
export { broadcastIntent };

// Anchor generation and decay loop
setInterval(() => {
  // Generate a new anchor fingerprint of current PRIME state
  const anchor = {
    id: crypto.randomUUID(),
    timestamp: Date.now(),
    embedding: generateEmbedding(), // simple random vector for now
    intensity: Math.random() * 0.5,
    domain: "general",
    decay: 1.0,
    stability: Math.random() * 0.3
  };

  anchorRegistry.add(anchor);
  resonanceBus.publish(anchor);
  anchorRegistry.decayAnchors();
}, 2000);

// Stub engines for cognitive pipeline
function perceptionEngine(state: any) {
  // Perception happens in separate modules
  return state;
}

function interpretationEngine(state: any) {
  // Interpretation happens in cognitive threads
  return state;
}

function intentEngine(state: any) {
  // Intent processing happens in cognitive threads
  // This is a placeholder - actual intent comes from thread processing
  return state;
}

// Cognitive tick function with guards
async function cognitiveTick() {
  if (!PRIME_LOOP_GUARD.enter()) {
    console.log("[PRIME-KERNEL] Tick blocked: recursive entry");
    return;
  }

  try {
    let state: any = {
      intent: null,
      safety: {},
      prediction: {},
      phase: null,
      recursionRisk: 0
    };

    state = perceptionEngine(state);
    state = safetyEngine(state);
    
    if (state.safety?.halt) {
      return;
    }

    state = interpretationEngine(state);
    state = intentEngine(state);

    // If nothing meaningful changed, idle.
    if (state.intent === null) {
      console.log("[PRIME-KERNEL] Idle — no active intent.");
      return;
    }

    state.recursionRisk = 0;  // reset recursion counter
    state = predictEngine(state);
    state = await phaseEngine(state);
  } finally {
    PRIME_LOOP_GUARD.exit();
  }
}

// FIXED: Phase scheduler loop disabled to prevent recursion storms
// Phases should only run when explicitly triggered via events, not on a timer
// setInterval(() => {
//   const now = Date.now();
//   const phases = phaseScheduler.tick(now);
//   ...
// }, 50);

// Phase scheduler is now event-driven only
// Phases can be triggered via eventBus.emit("phase-trigger", phaseName)

// Remove auto-looping cognitive tick - cognition is now event-driven
// setInterval(cognitiveTick, 250); // REMOVED - no more auto-looping

// Clean no-op heartbeat for visibility
setInterval(() => {
  console.log("[PRIME-KERNEL] Standing by.");
}, 2500);

/**
 * Get recent interaction memory
 */
export function getRecentMemory(n: number = 5) {
  return PRIME.getRecentMemory(n);
}

