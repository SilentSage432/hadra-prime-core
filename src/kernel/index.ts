// HADRA-PRIME Kernel Entry Point
// This file simply boots PRIME Core and exposes no internal modules.

import PRIME from "../prime.ts"; // triggers PRIME initialization + log emit

console.log("[KERNEL] HADRA-PRIME core boot sequence complete.");

/**
 * Get recent interaction memory
 */
export function getRecentMemory(n: number = 5) {
  return PRIME.getRecentMemory(n);
}

