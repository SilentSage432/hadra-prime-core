import { StabilityMatrix } from "../stability/stability_matrix.ts";
import { SafetyGuard } from "../safety/safety_guard.ts";
import { ThreadPool } from "./threads/thread_pool.ts";
import { TemporalRing } from "../memory/temporal_ring.ts";
import { DistributedState } from "../distributed/state_snapshot.ts";
import { ClusterBus } from "../distributed/cluster_bus.ts";

export function startRuntime(modules: any) {
  console.log("⏱️ HADRA-PRIME Runtime Scheduler Activated");

  // Initialize thread pool if not already initialized
  if (ThreadPool.threads.length === 0) {
    ThreadPool.init();
  }

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

    // Broadcast distributed snapshot for future cluster sync
    const distributedSnapshot = DistributedState.getSnapshot();
    ClusterBus.broadcast(distributedSnapshot);
  }, 3000);
}

/**
 * Dispatch a cognitive task to the thread pool
 */
export async function dispatchCognitiveTask(eventPayload: any) {
  const result = await ThreadPool.dispatch(eventPayload);
  if (result) {
    // Write temporal event after cognition cycle produces output
    const snapshot = StabilityMatrix.getSnapshot();
    TemporalRing.push({
      ts: Date.now(),
      input: eventPayload,
      fused: result.fused,
      intent: result.intent,
      output: result.output,
      stabilityScore: snapshot?.score,
    });

    // Broadcast distributed snapshot after thread processing
    const distributedSnapshot = DistributedState.getSnapshot();
    ClusterBus.broadcast(distributedSnapshot);

    // Emit result for downstream processing
    // Future: integrate with event system
    return result;
  }
  return null;
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

