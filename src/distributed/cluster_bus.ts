// src/distributed/cluster_bus.ts

import type { DistributedSnapshot } from "./state_snapshot.ts";

type Listener = (snapshot: DistributedSnapshot) => void;

export class ClusterBus {
  private static listeners: Listener[] = [];

  static onSnapshot(listener: Listener) {
    this.listeners.push(listener);
  }

  static broadcast(snapshot: DistributedSnapshot) {
    // Real networking comes later in P-CORE-B
    for (const l of this.listeners) l(snapshot);
  }
}

