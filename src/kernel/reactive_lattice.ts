// src/kernel/reactive_lattice.ts

import { PredictiveCoherence } from "../cognition/predictive_coherence.ts";
import { harmonizationEngine } from "../intent_engine/harmonization.ts";
import { anchorRegistry } from "../memory/index.ts";
import type { MemoryAnchor } from "../memory/anchors.ts";

export interface MicroInsight {
  type: string;
  weight: number;
  detail?: string;
  timestamp: number;
}

// Simple prediction engine stub for drift calculation
class PredictionEngineStub {
  private lastPrediction: number = 0.5;

  getDrift(): number {
    const current = PredictiveCoherence.computeConsensus().recursionRisk;
    const drift = current - this.lastPrediction;
    this.lastPrediction = current;
    return drift;
  }
}

// Simple memory engine stub for field strength
class MemoryEngineStub {
  getFieldStrength(): number {
    // Calculate based on anchor registry activity
    const anchorList = anchorRegistry.list();
    if (anchorList.length === 0) return 0.1;
    
    // Field strength based on recent anchor activity and decay
    const now = Date.now();
    const recentAnchors = anchorList.filter((a: MemoryAnchor) => 
      now - a.timestamp < 10000
    );
    return Math.min(1, recentAnchors.length / 5);
  }
}

const predictionEngine = new PredictionEngineStub();
const memoryEngine = new MemoryEngineStub();

export class ReactiveLattice {
  private insights: MicroInsight[] = [];
  private dampen = 0;

  // Fire small "micro threads"
  runMicrocycles() {
    // Dampener prevents reactive bursts from fueling loops
    if (this.dampen > 0) {
      this.dampen -= 1;
      return;
    }
    
    // FIXED: Add guard to prevent recursion storms
    // Only run microcycles when explicitly requested (not from auto-loops)
    if (!(globalThis as any).__PRIME_ALLOW_MICROCYCLES) {
      return;
    }
    
    // Prediction drift
    const drift = predictionEngine.getDrift();
    if (Math.abs(drift) > 0.2) {
      this.addInsight({
        type: "PREDICTION_DRIFT",
        weight: drift,
        detail: "High drift detected",
        timestamp: Date.now()
      });
    }

    // Memory resonance health
    const mem = memoryEngine.getFieldStrength();
    if (mem < 0.2) {
      this.addInsight({
        type: "WEAK_MEMORY_FIELD",
        weight: mem,
        timestamp: Date.now()
      });
    }

    // Harmonization alignment
    const align = harmonizationEngine.getRecentAlignment?.() ?? 0.5;
    if (align < 0.3) {
      this.addInsight({
        type: "HARMONIZATION_LOSS",
        weight: align,
        timestamp: Date.now()
      });
    }

    // Cool-off bad conditions
    if (Math.abs(drift) > 0.4 || align < 0.2) {
      this.addInsight({
        type: "COOLING_TRIGGER",
        weight: 0.7,
        timestamp: Date.now()
      });
      // If heavy signals detected, activate cooldown
      if (Math.abs(drift) > 0.3 || align < 0.3) {
        this.dampen = 5; // skip next 5 cycles
      }
    }
  }

  addInsight(insight: MicroInsight) {
    this.insights.push(insight);
    if (this.insights.length > 50) this.insights.shift();
  }

  getInsights() {
    return this.insights;
  }
}

export const reactiveLattice = new ReactiveLattice();

