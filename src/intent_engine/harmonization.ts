// src/intent_engine/harmonization.ts

export interface IntentSignature {
  id: string;
  timestamp: number;
  category: string;
  purpose: string;
  // 0-1 confidence of local interpretation
  confidence: number;
  // 16d vector fingerprint
  vector: number[];
  // urgency or importance
  weight: number;
}

export interface HarmonizedIntent {
  local: IntentSignature;
  alignment: number;     // 0-1 cluster consensus
  adjustedConfidence: number;
}

type Listener = (sig: IntentSignature) => void;

class HarmonizationBus {
  private listeners: Listener[] = [];

  publish(sig: IntentSignature) {
    for (const l of this.listeners) l(sig);
  }

  subscribe(listener: Listener) {
    this.listeners.push(listener);
  }
}

export const harmonizationBus = new HarmonizationBus();

export class HarmonizationEngine {
  private recent: IntentSignature[] = [];

  ingest(sig: IntentSignature) {
    this.recent.push(sig);
    if (this.recent.length > 50) this.recent.shift();
  }

  harmonize(local: IntentSignature): HarmonizedIntent {
    if (this.recent.length === 0) {
      return {
        local,
        alignment: 0.5,
        adjustedConfidence: local.confidence * 0.9
      };
    }

    const sims = this.recent.map(r => cosine(local.vector, r.vector));
    const avgSim = sims.reduce((a, b) => a + b, 0) / sims.length;

    const adjusted = (local.confidence * 0.7) + (avgSim * 0.3);

    return {
      local,
      alignment: avgSim,
      adjustedConfidence: adjusted
    };
  }

  getRecentAlignment(): number {
    if (this.recent.length === 0) return 0.5;
    const last = this.recent[this.recent.length - 1];
    return last.confidence; // proxy for now
  }
}

export const harmonizationEngine = new HarmonizationEngine();

function cosine(a: number[], b: number[]): number {
  const dot = a.reduce((sum, val, i) => sum + val * (b[i] || 0), 0);
  const normA = Math.sqrt(a.reduce((s, v) => s + v * v, 0));
  const normB = Math.sqrt(b.reduce((s, v) => s + v * v, 0));
  return dot / (normA * normB || 1);
}

