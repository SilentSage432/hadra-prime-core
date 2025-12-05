// src/distributed/state_snapshot.ts
// A124: Extended with uncertainty sharing for dual-mind coordination

import { metaEngine } from "../meta/meta_engine.ts";
import { NodeIdentity } from "./node_identity.ts";

export interface DistributedSnapshot {
  node: ReturnType<typeof NodeIdentity.getIdentity>;
  state: ReturnType<typeof metaEngine.getState>;
  timestamp: number;
  uncertainty?: number;  // A124: Uncertainty score for dual-mind awareness
}

export class DistributedState {
  static getSnapshot(cognitiveState?: any): DistributedSnapshot {
    const snapshot: DistributedSnapshot = {
      node: NodeIdentity.getIdentity(),
      state: metaEngine.getState(),
      timestamp: Date.now(),
    };

    // A124: Add uncertainty score if available in cognitive state
    if (cognitiveState?.uncertaintyScore !== undefined) {
      snapshot.uncertainty = cognitiveState.uncertaintyScore;
    }

    return snapshot;
  }
}

