// ESM requires explicit .ts extension
import PRIME, { getStatus, processCommand, subscribeLogs } from "./prime.ts";
import { kernelInstance } from "./kernel/index.ts";
import type { OperatorCommand } from "./operator/command_protocol.ts";

// Export default PRIME instance
export default PRIME;

// Export named functions for gateway compatibility
export { getStatus, processCommand, subscribeLogs };

// A48: Export operator command API
export function sendOperatorCommand(cmd: OperatorCommand) {
  return kernelInstance.handleOperatorCommand(cmd);
}

// Export expression types
export * from "./expression/types.ts";

// Export Synthetic Emotion Layer
export { SEL } from "./emotion/sel.ts";

// A64: Export Strategy Engine
export { StrategyEngine } from "./strategy/index.ts";

// A92: Export Embedding Adapter
export { getEmbeddingAdapter } from "./kernel/index.ts";

