// src/neural/cortex/cortex_manager.ts
// A91: Neural Cortex Manager
// Loads neural models from /models, validates they obey the NIC, provides standard API

import { NIC } from "../contract/neural_interaction_contract.ts";

export class CortexManager {
  private models: Map<string, any> = new Map();
  private registry: any[] = [];

  constructor(registry: any[]) {
    this.registry = registry;
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

