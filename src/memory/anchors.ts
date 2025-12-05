// src/memory/anchors.ts

export interface MemoryAnchor {
  id: string;
  timestamp: number;
  // semantic vector (abstract shape only)
  embedding: number[];
  // urgency / emotional-like stabilizer
  intensity: number;
  // classification hint
  domain: string;
  // natural decay scalar
  decay: number;
  // predictive stabilizer imported from prediction engine
  stability: number;
}

export class AnchorRegistry {
  private anchors: MemoryAnchor[] = [];

  add(anchor: MemoryAnchor) {
    this.anchors.push(anchor);
  }

  list() {
    return this.anchors;
  }

  decayAnchors() {
    const now = Date.now();
    this.anchors = this.anchors
      .map(a => ({
        ...a,
        decay: Math.max(0, a.decay - ((now - a.timestamp) / 10000))
      }))
      .filter(a => a.decay > 0.05);
  }
}

export const anchorRegistry = new AnchorRegistry();

