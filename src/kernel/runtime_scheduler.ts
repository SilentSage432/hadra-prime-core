import { StabilityMatrix } from "../stability/stability_matrix.ts";
import { SafetyGuard } from "../safety/safety_guard.ts";
import { ThreadPool } from "./threads/thread_pool.ts";
import { TemporalRing } from "../memory/temporal_ring.ts";
import { DistributedState } from "../distributed/state_snapshot.ts";
import { ClusterBus } from "../distributed/cluster_bus.ts";
import { DualSyncChannel } from "../distributed/dual_sync_channel.ts";
import { DualMind } from "../dual_core/dual_mind_activation.ts";

export function startRuntime(modules: any) {
  console.log("⏱️ HADRA-PRIME Runtime Scheduler Activated");

  // Initialize thread pool if not already initialized
  if (ThreadPool.threads.length === 0) {
    ThreadPool.init();
  }

  // A104b: Dual-Mind integration
  if (DualMind.isActive()) {
    DualSyncChannel.broadcast("runtime_tick", {});
  }

  // FIXED: Disabled auto-looping runtime scheduler to prevent recursion storms
  // Runtime monitoring should be event-driven, not on a timer
  // setInterval(() => {
  //   // Pre-cognition safety check
  //   if (!SafetyGuard.preCognitionCheck()) {
  //     console.warn("[PRIME] Cognition cycle skipped due to safety constraints.");
  //     return;
  //   }
  //   const snapshot = StabilityMatrix.getSnapshot();  // ← TRIGGERS PREDICTIONS
  //   const distributedSnapshot = DistributedState.getSnapshot();
  //   ClusterBus.broadcast(distributedSnapshot);
  // }, 3000);
  
  console.log("[PRIME-RUNTIME] Runtime scheduler disabled — operating in event-driven mode");
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

