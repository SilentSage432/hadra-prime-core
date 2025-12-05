// A112 — Temporal-State Embedding Model
// Neural Cortex Slot #4: Temporal-State Embedding Model
// Encodes PRIME's temporal cognition: sequences, evolving states, time-aware structure

import type { TemporalSnapshot } from "../../temporal/window.ts";
import type { NeuralInteractionPayload, TemporalStateWindow } from "../contract/neural_interaction_contract.ts";

// Re-export for convenience
export type { TemporalStateWindow, NeuralInteractionPayload };

export class TemporalEmbeddingModel {
  modelName = "temporal_state_encoder_v1";
  isReady = false;

  async warmup() {
    // Placeholder for PyTorch warmup; stub for now
    // In A120+, this will load actual PyTorch model weights
    this.isReady = true;
    console.log("[PRIME-NEURAL] Temporal Embedding Model warmed up (stub mode).");
  }

  encodeSequence(window: TemporalStateWindow): number[] {
    if (!this.isReady) {
      console.warn("[PRIME-NEURAL] Temporal model not ready, returning empty embedding.");
      return [];
    }

    if (!window.states || window.states.length === 0) {
      return [];
    }

    // Simple deterministic stub → later replaced with tensor embedding
    // This creates a stable embedding based on state sequence
    const embedding: number[] = [];

    window.states.forEach((state, i) => {
      // Hash-based deterministic encoding
      const hash = [...state.description || ""]
        .map((c) => c.charCodeAt(0))
        .reduce((a, b) => a + b, 0);

      // Normalize hash to 0-1 range
      const normalizedHash = (hash % 997) / 997;
      
      // Add temporal position encoding
      const positionEncoding = i * 0.0001;
      
      // Encode state values
      const stateVector = [
        normalizedHash + positionEncoding,
        state.clarity || 0,
        state.consolidation || 0,
        state.curiosity || 0,
        state.stability || 0,
        (state.t || 0) % 1000 / 1000 // Normalize timestamp
      ];

      embedding.push(...stateVector);
    });

    // Pad or truncate to fixed size (64 dimensions for now)
    const targetSize = 64;
    if (embedding.length < targetSize) {
      embedding.push(...new Array(targetSize - embedding.length).fill(0));
    } else if (embedding.length > targetSize) {
      return embedding.slice(0, targetSize);
    }

    return embedding;
  }

  compute(payload: NeuralInteractionPayload): number[] {
    if (!this.isReady) return [];

    if (payload.type !== "temporal_sequence") {
      console.warn("[PRIME-NEURAL] Temporal model received non-temporal payload type:", payload.type);
      return [];
    }

    if (!payload.temporalWindow) {
      console.warn("[PRIME-NEURAL] Temporal payload missing temporalWindow.");
      return [];
    }

    return this.encodeSequence(payload.temporalWindow);
  }

  // A112: Adaptive boundary check for temporal sequences
  adaptiveBoundaryCheck(window: TemporalStateWindow): boolean {
    if (!window.states || window.states.length === 0) return false;
    
    // Check sequence length bounds
    if (window.states.length > 1000) {
      console.warn("[PRIME-NEURAL] Temporal window too large, truncating.");
      return false;
    }

    // Check temporal span (max 1 hour)
    if (window.states.length > 1) {
      const first = window.states[0].t;
      const last = window.states[window.states.length - 1].t;
      const span = last - first;
      if (span > 3600000) { // 1 hour in ms
        console.warn("[PRIME-NEURAL] Temporal window span too large.");
        return false;
      }
    }

    return true;
  }
}

