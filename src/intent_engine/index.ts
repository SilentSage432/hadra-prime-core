/**
 * Intent Engine Module
 * Handles goal selection, decision matrix, and action routing
 */

import { IntentEngine } from "./intent_engine.ts";

// Export harmonization module
export * from "./harmonization.ts";

export interface PrimeContext {
  emit: (event: string, ...args: any[]) => boolean;
  log: (message: string) => void;
  getStatus: () => any;
  remember: (event: any) => void;
  store: (topic: string, entry: any) => void;
  recall: (topic: string) => any[];
  recallRecent: () => any[];
}

export default class IntentEngineModule {
  private engine: IntentEngine;

  constructor() {
    this.engine = new IntentEngine();
  }

  init(context: PrimeContext) {
    context.log("Intent engine module initialized");
  }

  process(input: string) {
    return this.engine.process(input);
  }
}

