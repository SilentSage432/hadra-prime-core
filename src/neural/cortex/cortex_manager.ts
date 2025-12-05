// src/neural/cortex/cortex_manager.ts
// A91: Neural Cortex Manager
// A112: Expanded with Slot #4 (Temporal Embedding Model)
// Loads neural models from /models, validates they obey the NIC, provides standard API

import { NIC, type NeuralPacket } from "../contract/neural_interaction_contract.ts";
import { TemporalEmbeddingModel } from "../models/temporal_embedding_model.ts";
import { NeuralContextEncoder } from "../context_encoder.ts";
import type { SharedEmbedding } from "../../shared/embedding.ts";

export class CortexManager {
  private models: Map<string, any> = new Map();
  private registry: any[] = [];
  
  // A112: Neural slots system
  private slots: Record<number, any> = {
    1: null,
    2: null,
    3: null,
    4: new TemporalEmbeddingModel(), // A112: Slot #4 - Temporal Embedding Model
  };

  // A112: External signal ingestion for SAGE-PRIME convergence (NCI)
  // A113: Shared embedding integration
  private externalInputs: NeuralPacket[] = [];
  private contextIntegration: any[] = [];
  private sharedEmbeddings: SharedEmbedding[] = [];
  private encoder: NeuralContextEncoder;

  constructor(registry: any[]) {
    this.registry = registry;
    this.encoder = new NeuralContextEncoder();
  }

  // A112: Initialize all slots
  async initialize() {
    for (const id of Object.keys(this.slots)) {
      const slot = this.slots[Number(id)];
      if (slot && slot.warmup) {
        await slot.warmup();
        console.log(`[CORTEX] Slot #${id} initialized: ${slot.modelName || slot.constructor.name}`);
      }
    }
  }

  // A112: Get slot by ID
  getSlot(id: number): any {
    return this.slots[id] || null;
  }

  // A112: Load model into slot
  async loadModelIntoSlot(id: number, model: any) {
    if (id < 1 || id > 4) {
      console.warn(`[CORTEX] Invalid slot ID: ${id}`);
      return false;
    }
    
    if (model && model.warmup) {
      await model.warmup();
    }
    
    this.slots[id] = model;
    console.log(`[CORTEX] Model loaded into slot #${id}`);
    return true;
  }

  async loadModel(name: string) {
    const entry = this.registry.find(r => r.name === name && r.enabled);

    if (!entry) {
      console.warn(`[CORTEX] No enabled model named '${name}'.`);
      return false;
    }

    try {
      const module = await import(entry.path);

      if (!module?.infer) {
        console.warn(`[CORTEX] '${name}' does not export infer().`);
        return false;
      }

      this.models.set(name, module);
      console.log(`[CORTEX] Loaded neural model '${name}'.`);
      return true;
    } catch (err) {
      console.error(`[CORTEX] Failed to load '${name}':`, err);
      return false;
    }
  }

  unloadModel(name: string) {
    if (this.models.has(name)) {
      this.models.delete(name);
      console.log(`[CORTEX] Unloaded model '${name}'.`);
    }
  }

  async reloadModel(name: string) {
    this.unloadModel(name);
    return await this.loadModel(name);
  }

  isLoaded(name: string) {
    return this.models.has(name);
  }

  async infer(name: string, input: any) {
    if (!NIC.validateInput(input)) {
      console.warn("[CORTEX] NIC input contract rejected request.");
      return null;
    }

    const model = this.models.get(name);
    if (!model) {
      console.warn(`[CORTEX] Model '${name}' not loaded.`);
      return null;
    }

    const output = await model.infer(input);

    if (!NIC.validateOutput(output)) {
      console.warn("[CORTEX] NIC output contract rejected neural output.");
      return null;
    }

    return output;
  }

  // A112: Ingest external neural signals from SAGE (NCI)
  ingestExternalSignal(packet: NeuralPacket): void {
    // SAGE → PRIME neural ingestion
    this.externalInputs.push(packet);
    
    // Maintain bounded history
    if (this.externalInputs.length > 1000) {
      this.externalInputs.shift();
    }

    // Convert raw embedding into PRIME context features
    try {
      const ctx = this.encodeExternalEmbedding(packet.embedding, packet.contextVector);
      this.contextIntegration.push(ctx);
      
      // Maintain bounded integration history
      if (this.contextIntegration.length > 500) {
        this.contextIntegration.shift();
      }

      console.log("[PRIME-NCI] Ingested external neural signal:", {
        source: packet.source,
        channel: packet.channel,
        embeddingSize: packet.embedding.length,
        contextSize: packet.contextVector.length
      });
    } catch (error) {
      console.error("[PRIME-NCI] Error encoding external embedding:", error);
    }
  }

  /**
   * Encode external embedding (from SAGE) into PRIME context features
   */
  private encodeExternalEmbedding(embedding: number[], contextVector: number[]): any {
    // Normalize embedding to PRIME's expected format
    const normalizedEmbedding = embedding.slice(0, 64); // Ensure 64-dim vector
    while (normalizedEmbedding.length < 64) {
      normalizedEmbedding.push(0);
    }

    // Combine embedding and context vector
    const combined = [...normalizedEmbedding, ...contextVector.slice(0, 32)];
    
    return {
      vector: combined.slice(0, 64), // Ensure 64-dim output
      source: "external",
      timestamp: Date.now(),
      originalEmbedding: embedding,
      originalContext: contextVector
    };
  }

  /**
   * Get recent external inputs (SAGE signals)
   */
  getRecentExternalInputs(count: number = 10): NeuralPacket[] {
    return this.externalInputs.slice(-count);
  }

  /**
   * Get integrated context from external signals
   */
  getIntegratedContext(): any[] {
    return [...this.contextIntegration];
  }

  /**
   * A113: Ingest shared embedding (from PRIME or SAGE)
   */
  ingestSharedEmbedding(embedding: SharedEmbedding): void {
    this.sharedEmbeddings.push(embedding);
    
    // Maintain bounded history
    if (this.sharedEmbeddings.length > 1000) {
      this.sharedEmbeddings.shift();
    }

    // Also add to context integration
    this.contextIntegration.push({
      vector: embedding.vector,
      source: embedding.origin,
      signalType: embedding.signalType,
      epoch: embedding.epoch,
      timestamp: Date.now()
    });

    // Maintain bounded integration history
    if (this.contextIntegration.length > 500) {
      this.contextIntegration.shift();
    }

    console.log("[PRIME-NCI] Ingested shared embedding:", {
      origin: embedding.origin,
      signalType: embedding.signalType,
      vectorSize: embedding.vector.length,
      epoch: embedding.epoch
    });
  }

  /**
   * A113: Get recent shared embeddings
   */
  getRecentSharedEmbeddings(count: number = 10): SharedEmbedding[] {
    return this.sharedEmbeddings.slice(-count);
  }

  /**
   * A113: Get shared embeddings by origin
   */
  getSharedEmbeddingsByOrigin(origin: "PRIME" | "SAGE"): SharedEmbedding[] {
    return this.sharedEmbeddings.filter(e => e.origin === origin);
  }

  /**
   * A113: Get shared embeddings by signal type
   */
  getSharedEmbeddingsBySignalType(
    signalType: "cognition" | "state" | "intent" | "emotion"
  ): SharedEmbedding[] {
    return this.sharedEmbeddings.filter(e => e.signalType === signalType);
  }
}

