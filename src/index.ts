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

// A112: Export Neural Convergence Interface (NCI)
export { NeuralBridge, encodePrimeEmbedding, decodeSageEmbedding } from "./neural/neural_bridge.ts";
export { NeuralConvergenceContract, type NeuralPacket } from "./neural/contract/neural_interaction_contract.ts";
// A113: Export Neural Embedding Pipeline
export { primeEmbeddingAdapter } from "./neural/embedding/prime_embedding_adapter.ts";
export { sageEmbeddingParser } from "./neural/embedding/sage_embedding_parser.ts";
export type { SharedEmbedding } from "./shared/embedding.ts";
// A114: Export Cross-Mind Neural Synchronization
export { DualMindSyncManager } from "./dual_mind/sync_manager.ts";
export { NeuralSyncEngine } from "./dual_mind/synchronization_engine.ts";
export { ingestFederationPacket, createSagePulsePacket } from "./perception/federation_ingest.ts";
// A116: Export Joint Situation Modeler
export { JointSituationModeler, type JointSituationSnapshot } from "./situation_model/joint_situation_modeler.ts";

