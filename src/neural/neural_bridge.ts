// src/neural/neural_bridge.ts
// A101: Neural Slot #2 Preparation Layer
// A112: Neural Bridge v2 - Tensor Handshake Layer with temporal sequence support
// A112: Neural Convergence Interface (NCI) - SAGE-PRIME neural handshake
// Cross-Modal Embedding Bridge
// This is the Rosetta Stone between PRIME's rule-based engine and the neural substrate
// A108b: Hardware-aware neural offloading
// A127: Dual-Mind Conflict Resolution integration

import type { SymbolicPacket } from "../shared/symbolic_packet.ts";
import type { HardwareProfile } from "../hardware/hardware_profile.ts";
import type { NeuralInteractionPayload } from "./models/temporal_embedding_model.ts";
import type { NeuralPacket } from "./contract/neural_interaction_contract.ts";
import { NeuralConvergenceContract } from "./contract/neural_interaction_contract.ts";
import { primeEmbeddingAdapter } from "./embedding/prime_embedding_adapter.ts";
import { sageEmbeddingParser } from "./embedding/sage_embedding_parser.ts";
import type { SharedEmbedding } from "../shared/embedding.ts";

export interface TensorStub {
  shape: number[];
  data: number[];
}

export class NeuralBridge {
  private static hardware: HardwareProfile | null = null;

  static init() {
    this.hardware = (globalThis as any).__PRIME_HARDWARE__ || null;
    // A127: Log conflict evaluation capability
    console.log("[PRIME↔SAGE] Conflict evaluation enabled.");
  }

  static shouldOffloadNeuralOps(): boolean {
    if (!this.hardware) {
      this.hardware = (globalThis as any).__PRIME_HARDWARE__ || null;
    }
    return this.hardware?.neuralCapacity === "tiny" || this.hardware?.neuralCapacity === "none";
  }

  static canRunNeuralLocally(): boolean {
    if (!this.hardware) {
      this.hardware = (globalThis as any).__PRIME_HARDWARE__ || null;
    }
    const adaptiveConfig = (globalThis as any).__PRIME_ADAPTIVE_CONFIG__;
    return adaptiveConfig?.enableNeuralSlots === true && 
           this.hardware?.neuralCapacity !== "tiny" && 
           this.hardware?.neuralCapacity !== "none";
  }
  // Converts symbolic representations into a lightweight tensor stub
  static encodeToTensor(symbolic: SymbolicPacket): TensorStub {
    const text = JSON.stringify(symbolic);
    return {
      shape: [1, text.length],
      data: text.split("").map(c => c.charCodeAt(0) / 255)
    };
  }

  // Converts a tensor stub back into symbolic data
  static decodeFromTensor(tensor: TensorStub): SymbolicPacket {
    try {
      const str = tensor.data
        .map(v => String.fromCharCode(Math.floor(v * 255)))
        .join("");
      return JSON.parse(str) as SymbolicPacket;
    } catch (e) {
      return { type: "unknown", payload: { error: "decode_failed" } };
    }
  }

  // Future neural hook — for PyTorch model inference
  static async runNeuralModel(tensor: TensorStub): Promise<TensorStub> {
    // A108b: Check if we should offload to SAGE server
    if (this.shouldOffloadNeuralOps()) {
      console.log("[PRIME-NEURAL] Offloading neural operation to SAGE server (hardware constraint).");
      // TODO: Implement SAGE offloading protocol
      // For now, return stub
      return tensor;
    }

    // Check if we can run locally
    if (!this.canRunNeuralLocally()) {
      console.log("[PRIME-NEURAL] Neural operations disabled (hardware constraint).");
      return tensor; // Return stub
    }

    // Stub: this is where PyTorch inference will run after A120+
    return tensor; // Echo back until neural engine exists
  }

  // A112: Compute neural embeddings/dispatch to appropriate slot
  static compute(payload: NeuralInteractionPayload): number[] {
    // A112: Dispatch temporal sequences to Slot #4
    if (payload.type === "temporal_sequence") {
      const cortex = (globalThis as any).PRIME_CORTEX;
      if (!cortex) {
        console.warn("[PRIME-NEURAL] Cortex not available for temporal sequence computation.");
        return [];
      }

      const slot4 = cortex.getSlot(4);
      if (!slot4) {
        console.warn("[PRIME-NEURAL] Slot #4 (Temporal Embedding) not available.");
        return [];
      }

      // Validate temporal window
      if (slot4.adaptiveBoundaryCheck && payload.temporalWindow) {
        if (!slot4.adaptiveBoundaryCheck(payload.temporalWindow)) {
          console.warn("[PRIME-NEURAL] Temporal window failed boundary check.");
          return [];
        }
      }

      return slot4.compute(payload);
    }

    // Other payload types can be handled here in future
    console.warn("[PRIME-NEURAL] Unsupported payload type:", payload.type);
    return [];
  }

  // A112: Get singleton instance accessor
  static getInstance(): typeof NeuralBridge {
    return NeuralBridge;
  }

  // A112: Neural Convergence Interface (NCI) - SAGE-PRIME neural handshake
  private static listeners: ((packet: NeuralPacket) => void)[] = [];

  /**
   * Send neural packet from SAGE to PRIME
   * Validates and sanitizes before ingestion
   * A113: Converts to shared embedding format
   */
  static sendToPrime(packet: NeuralPacket): boolean {
    if (!NeuralConvergenceContract.validate(packet)) {
      console.warn("[PRIME-NCI] Invalid neural packet rejected:", packet.id);
      return false;
    }

    const clean = NeuralConvergenceContract.sanitize(packet);
    
    // A113: Convert SAGE packet to shared embedding
    const shared = sageEmbeddingParser.parse(clean);
    
    // Get cortex manager to ingest external signal and shared embedding
    const cortex = (globalThis as any).PRIME_CORTEX;
    if (cortex) {
      if (cortex.ingestExternalSignal) {
        cortex.ingestExternalSignal(clean);
      }
      if (cortex.ingestSharedEmbedding) {
        cortex.ingestSharedEmbedding(shared);
      }
      
      console.log("[PRIME-NCI] Ingested SAGE neural packet:", {
        id: clean.id,
        channel: clean.channel,
        embeddingSize: clean.embedding.length,
        signalType: shared.signalType
      });
      return true;
    } else {
      console.warn("[PRIME-NCI] Cortex not available for external signal ingestion");
      return false;
    }
  }

  /**
   * Register listener for PRIME neural events
   */
  static onPrimeNeuralEvent(callback: (packet: NeuralPacket) => void): () => void {
    this.listeners.push(callback);
    // Return unsubscribe function
    return () => {
      const index = this.listeners.indexOf(callback);
      if (index > -1) {
        this.listeners.splice(index, 1);
      }
    };
  }

  /**
   * Emit neural packet from PRIME (to SAGE or other listeners)
   * A113: Converts to shared embedding format before emission
   */
  static emitFromPrime(packet: NeuralPacket): void {
    if (!NeuralConvergenceContract.validate(packet)) {
      console.warn("[PRIME-NCI] Invalid PRIME neural packet, not emitting:", packet.id);
      return;
    }

    const clean = NeuralConvergenceContract.sanitize(packet);
    
    // A113: Convert PRIME embedding to shared format
    const shared = encodePrimeEmbedding(clean.embedding, clean.channel);
    
    // Emit both packet and shared embedding
    this.listeners.forEach(listener => {
      try {
        listener(clean);
        // Also emit shared embedding if listener accepts it
        if (typeof (listener as any).onSharedEmbedding === "function") {
          (listener as any).onSharedEmbedding(shared);
        }
      } catch (error) {
        console.error("[PRIME-NCI] Error in neural event listener:", error);
      }
    });

    console.log("[PRIME-NCI] Emitted PRIME neural packet:", {
      id: clean.id,
      channel: clean.channel,
      source: clean.source,
      signalType: shared.signalType
    });
  }

  /**
   * A113: Encode PRIME embedding to shared embedding format
   */
  static encodePrimeEmbedding(
    vec: number[],
    channel: "memory" | "state" | "intent" | "perception" = "memory"
  ): SharedEmbedding {
    // Map channel to signal type
    let signalType: "cognition" | "state" | "intent" | "emotion" = "cognition";
    switch (channel) {
      case "state":
        signalType = "state";
        break;
      case "intent":
        signalType = "intent";
        break;
      case "perception":
        signalType = "cognition";
        break;
      case "memory":
        signalType = "cognition";
        break;
    }
    
    return primeEmbeddingAdapter.toSharedEmbedding(vec, signalType);
  }

  /**
   * A113: Decode SAGE neural packet to shared embedding format
   */
  static decodeSageEmbedding(packet: NeuralPacket): SharedEmbedding {
    return sageEmbeddingParser.parse(packet);
  }

  /**
   * A128: Send unified intent to SAGE
   * Makes the shared intent available to SAGE for coordination
   */
  static sendUnifiedIntent(intent: any): void {
    // Check if SAGE is available via global state
    const sageState = (globalThis as any).__SAGE_STATE__;
    if (!sageState) {
      console.log("[PRIME-NCI] SAGE not available for unified intent sharing.");
      return;
    }

    // Store unified intent in SAGE state for access
    if (sageState.receiveUnifiedIntent) {
      sageState.receiveUnifiedIntent(intent);
    } else {
      // Fallback: store in global state
      (globalThis as any).__UNIFIED_INTENT__ = intent;
    }

    console.log("[PRIME-NCI] Unified intent shared with SAGE:", {
      unifiedGoal: intent.unifiedGoal,
      unifiedUrgency: intent.unifiedUrgency?.toFixed(3),
      timestamp: intent.timestamp
    });
  }
}

// A113: Export convenience functions for embedding conversion
export function encodePrimeEmbedding(
  vec: number[],
  channel: "memory" | "state" | "intent" | "perception" = "memory"
): SharedEmbedding {
  return NeuralBridge.encodePrimeEmbedding(vec, channel);
}

export function decodeSageEmbedding(packet: NeuralPacket): SharedEmbedding {
  return NeuralBridge.decodeSageEmbedding(packet);
}

// A108b: Initialize hardware awareness on module load
NeuralBridge.init();

