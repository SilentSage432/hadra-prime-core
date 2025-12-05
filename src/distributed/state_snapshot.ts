// src/distributed/state_snapshot.ts

import { metaEngine } from "../meta/meta_engine.ts";
import { NodeIdentity } from "./node_identity.ts";

export interface DistributedSnapshot {
  node: ReturnType<typeof NodeIdentity.getIdentity>;
  state: ReturnType<typeof metaEngine.getState>;
  timestamp: number;
}

export class DistributedState {
  static getSnapshot(): DistributedSnapshot {
    return {
      node: NodeIdentity.getIdentity(),
      state: metaEngine.getState(),
      timestamp: Date.now(),
    };
  }
}

