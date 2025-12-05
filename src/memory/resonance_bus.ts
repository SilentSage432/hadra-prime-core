// src/memory/resonance_bus.ts

import type { MemoryAnchor } from "./anchors.ts";

type Listener = (anchor: MemoryAnchor) => void;

class ResonanceBus {
  private listeners: Listener[] = [];

  publish(anchor: MemoryAnchor) {
    for (const l of this.listeners) l(anchor);
  }

  subscribe(listener: Listener) {
    this.listeners.push(listener);
  }
}

export const resonanceBus = new ResonanceBus();

