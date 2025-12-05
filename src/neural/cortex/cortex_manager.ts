// src/neural/cortex/cortex_manager.ts
// A91: Neural Cortex Manager
// A112: Expanded with Slot #4 (Temporal Embedding Model)
// Loads neural models from /models, validates they obey the NIC, provides standard API

import { NIC } from "../contract/neural_interaction_contract.ts";
import { TemporalEmbeddingModel } from "../models/temporal_embedding_model.ts";

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

  constructor(registry: any[]) {
    this.registry = registry;
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
}

