// src/distributed/dual_sync_channel.ts
// A104b: Dual Sync Channel
// A105: Stabilized sync channel for SAGE <-> PRIME communication

import { EventEmitter } from "events";
import { CognitiveBoundaryFilter } from "./cognitive_boundary_filter.ts";
import { DualMindSafetyGate } from "../safety/dual_mind_safety_gate.ts";
import { NodeIdentity } from "./node_identity.ts";
import { SAGE_IDENTITY } from "./sage_identity.ts";

// PRIME identity helper
const PRIME_IDENTITY = {
  get id() {
    return NodeIdentity.getIdentity().nodeId;
  }
};

export const DualSyncChannel = new EventEmitter();

// Send PRIME -> SAGE (outbound)
export function sendToSAGE(payload: any) {
  const safePayload = CognitiveBoundaryFilter.validateOutbound(
    { ...payload, sender: PRIME_IDENTITY.id },
    "SAGE"
  );

  if (!safePayload) return;

  const safety = DualMindSafetyGate.checkBoundary("PRIME");
  if (!safety.allowed) {
    console.warn("[DUAL-SYNC] Blocked outbound to SAGE:", safety.reason);
    return;
  }

  DualSyncChannel.emit("prime_to_sage", safePayload);
}

// Receive SAGE -> PRIME (inbound)
DualSyncChannel.on("sage_to_prime", (payload: any) => {
  const safe = CognitiveBoundaryFilter.validateInbound(payload, "SAGE");
  if (!safe) {
    console.warn("[DUAL-SYNC] Blocked inbound from SAGE: validation failed");
    return;
  }

  const safety = DualMindSafetyGate.checkBoundary("SAGE");
  if (!safety.allowed) {
    console.warn("[DUAL-SYNC] Blocked inbound from SAGE:", safety.reason);
    return;
  }

  // Forward into PRIME cognition
  if (typeof process !== "undefined" && process.nextTick) {
    process.nextTick(() => {
      (globalThis as any).PRIME_EVENT_BUS?.emit("dual_mind:input", safe);
    });
  } else {
    // Fallback for non-Node environments
    setTimeout(() => {
      (globalThis as any).PRIME_EVENT_BUS?.emit("dual_mind:input", safe);
    }, 0);
  }
});

// Legacy broadcast method for compatibility
export class DualSyncChannelLegacy {
  static broadcast(event: string, payload: any) {
    console.log(`[DUAL-SYNC] ${event}`, payload);
    if (event === "runtime_tick") {
      sendToSAGE({ type: "runtime_tick", payload });
    }
  }
}

