// HADRA-PRIME Kernel Entry Point
// This file simply boots PRIME Core and exposes no internal modules.

import PRIME from "../prime.ts"; // triggers PRIME initialization + log emit
import { StabilityMatrix } from "../stability/stability_matrix.ts";
import { ThreadPool } from "./threads/thread_pool.ts";
import { anchorRegistry, resonanceBus } from "../memory/index.ts";
import { generateEmbedding } from "../shared/embedding.ts";
import crypto from "crypto";

console.log("[PRIME] Initializing Stability Matrix...");
StabilityMatrix.init();

console.log("[PRIME] Cognitive Fusion Stability Layer active.");

console.log("[PRIME] Initializing cognitive threads...");
// ThreadPool will be initialized with default instances
// For full integration, it should be initialized with PRIME's actual instances
ThreadPool.init();

console.log("[KERNEL] HADRA-PRIME core boot sequence complete.");

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

/**
 * Get recent interaction memory
 */
export function getRecentMemory(n: number = 5) {
  return PRIME.getRecentMemory(n);
}

