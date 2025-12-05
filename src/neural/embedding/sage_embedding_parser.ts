// A113 — SAGE Embedding Parser
// Converts SAGE neural packets into PRIME-readable shared embeddings
// Enables SAGE → PRIME neural communication

import type { NeuralPacket } from "../contract/neural_interaction_contract.ts";
import type { SharedEmbedding } from "../../shared/embedding.ts";

export class SageEmbeddingParser {
  /**
   * Parse SAGE neural packet into shared embedding format
   */
  parse(packet: NeuralPacket): SharedEmbedding {
    const base = packet.embedding;

    // Map SAGE channel to signal type
    let signalType: "cognition" | "state" | "intent" | "emotion" = "cognition";
    
    switch (packet.channel) {
      case "state":
        signalType = "state";
        break;
      case "intent":
        signalType = "intent";
        break;
      case "perception":
        signalType = "cognition"; // Perception maps to cognition in PRIME
        break;
      case "memory":
        signalType = "cognition"; // Memory maps to cognition in PRIME
        break;
      default:
        signalType = "cognition";
    }

    return {
      vector: base,
      origin: "SAGE",
      epoch: packet.timestamp,
      signalType
    };
  }

  /**
   * Batch parse multiple SAGE packets
   */
  parseBatch(packets: NeuralPacket[]): SharedEmbedding[] {
    return packets.map(packet => this.parse(packet));
  }

  /**
   * Extract signal type from SAGE packet metadata if available
   */
  private extractSignalType(packet: NeuralPacket): "cognition" | "state" | "intent" | "emotion" {
    // Check metadata for explicit signal type
    if (packet.metadata?.signalType) {
      const metaType = packet.metadata.signalType;
      if (["cognition", "state", "intent", "emotion"].includes(metaType)) {
        return metaType as "cognition" | "state" | "intent" | "emotion";
      }
    }

    // Fall back to channel mapping
    switch (packet.channel) {
      case "state":
        return "state";
      case "intent":
        return "intent";
      case "perception":
      case "memory":
      default:
        return "cognition";
    }
  }
}

export const sageEmbeddingParser = new SageEmbeddingParser();

