// A114 — Federation Packet Ingestion
// Handles ingestion of SAGE timing packets and other federation signals
// Enables cross-mind neural synchronization

import type { DualMindSyncManager } from "../dual_mind/sync_manager.ts";
import type { CrossMindAlignmentEngine } from "../cognition/prediction/cross_mind_alignment_engine.ts";

export interface FederationPacket {
  type: string;
  timestamp: number;
  source?: "SAGE" | "PRIME" | "FEDERATION";
  data?: any;
}

/**
 * Ingest federation packet (from SAGE or other federation nodes)
 * A114: Handles SAGE pulse timing for neural synchronization
 */
export function ingestFederationPacket(packet: FederationPacket): void {
  if (!packet || !packet.type) {
    console.warn("[PRIME-FEDERATION] Invalid packet received");
    return;
  }

  // A114: Handle SAGE pulse timing
  if (packet.type === "sage_pulse") {
    const sync = (globalThis as any).__PRIME_SYNC__ as DualMindSyncManager;
    if (sync) {
      sync.syncSagePulse(packet.timestamp);
      console.log(`[PRIME-SYNC] Sage pulse received (timestamp: ${packet.timestamp})`);
    } else {
      console.warn("[PRIME-SYNC] Sync manager not available");
    }
    return;
  }

  // A114: Handle SAGE neural state updates
  if (packet.type === "sage_neural_state") {
    const sync = (globalThis as any).__PRIME_SYNC__ as DualMindSyncManager;
    if (sync && packet.data?.pulseTime) {
      sync.syncSagePulse(packet.data.pulseTime);
    }
    return;
  }

  // A116: Handle SAGE situation updates
  if (packet.type === "sage_situation") {
    const JSM = (globalThis as any).__PRIME_JSM__;
    if (JSM) {
      // Get PRIME's current context
      const primeContext = (globalThis as any).__PRIME_CONTEXT__ || {
        timestamp: Date.now(),
        summary: "active cognition"
      };
      
      // Update JSM with SAGE situation data
      JSM.update(primeContext, packet.data || packet);
      console.log("[PRIME-JSM] Updated from SAGE situation packet");
    }
    return;
  }

  // A117: Handle SAGE state updates for cross-mind alignment
  if (packet.type === "sage_state_update" || packet.type === "sage_state") {
    const alignment = (globalThis as any).__PRIME_CROSS_MIND_ALIGNMENT__ as CrossMindAlignmentEngine;
    if (alignment) {
      const sageState = packet.data || packet;
      alignment.updateSageState({
        stability: sageState.stability || sageState.health || 0.7,
        motivation: sageState.motivation || 0.5,
        intentDirection: sageState.intentDirection || sageState.currentObjective || "neutral",
        cognitiveLoad: sageState.cognitiveLoad || sageState.load || 0.5,
        clusterState: sageState.clusterState || sageState
      });
      console.log("[PRIME-ALIGNMENT] Updated SAGE state for predictive alignment");
    }
    return;
  }

  // Handle other federation packet types
  console.log(`[PRIME-FEDERATION] Received packet: ${packet.type}`, {
    timestamp: packet.timestamp,
    source: packet.source
  });
}

/**
 * Create a SAGE pulse packet (for testing/integration)
 */
export function createSagePulsePacket(timestamp?: number): FederationPacket {
  return {
    type: "sage_pulse",
    timestamp: timestamp || Date.now(),
    source: "SAGE"
  };
}

