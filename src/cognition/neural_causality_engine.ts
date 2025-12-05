// A114 — Neural Causality Engine (NCE)
// PRIME learns cause → effect relationships from neural signals over time
// This is where PRIME develops Neural Causality Awareness

import type { NeuralSignal, CoherenceResult } from "./neural_symbolic_coherence.ts";

export interface CausalRecord {
  signal: NeuralSignal;
  coherence: CoherenceResult;
  effect: string;
  timestamp: number;
  context?: Record<string, any>; // Additional context for causal analysis
}

export interface CausalMap {
  [tag: string]: number; // Tag -> average relevance/influence level
}

export class NeuralCausalityEngine {
  private history: CausalRecord[] = [];
  private maxHistorySize = 2000;

  /**
   * Record a causal relationship: neural signal + coherence → effect
   */
  record(
    signal: NeuralSignal,
    coherence: CoherenceResult,
    effect: string,
    context?: Record<string, any>
  ) {
    this.history.push({
      signal,
      coherence,
      effect,
      timestamp: Date.now(),
      context
    });

    // Maintain bounded history
    if (this.history.length > this.maxHistorySize) {
      this.history.shift();
    }
  }

  /**
   * Infer causality patterns from historical records
   * Groups by semantic tags and computes average influence levels
   */
  inferCausality(): CausalMap {
    if (this.history.length === 0) {
      return {};
    }

    // Group by semantic tag
    const groups: Record<string, number[]> = {};

    for (const record of this.history) {
      const tags = record.coherence.semanticTags;
      
      for (const tag of tags) {
        if (!groups[tag]) {
          groups[tag] = [];
        }
        groups[tag].push(record.coherence.relevance);
      }
    }

    // Compute average relevance for each tag
    const causalMap: CausalMap = {};

    for (const tag of Object.keys(groups)) {
      const relevances = groups[tag];
      const avg = relevances.reduce((a, b) => a + b, 0) / relevances.length;
      causalMap[tag] = avg;
    }

    return causalMap;
  }

  /**
   * Get causal patterns for a specific effect type
   */
  inferCausalityForEffect(effect: string): CausalMap {
    const filtered = this.history.filter(r => r.effect === effect);
    
    if (filtered.length === 0) {
      return {};
    }

    const groups: Record<string, number[]> = {};

    for (const record of filtered) {
      const tags = record.coherence.semanticTags;
      
      for (const tag of tags) {
        if (!groups[tag]) {
          groups[tag] = [];
        }
        groups[tag].push(record.coherence.relevance);
      }
    }

    const causalMap: CausalMap = {};

    for (const tag of Object.keys(groups)) {
      const relevances = groups[tag];
      const avg = relevances.reduce((a, b) => a + b, 0) / relevances.length;
      causalMap[tag] = avg;
    }

    return causalMap;
  }

  /**
   * Get recent causal trends (last N records)
   */
  getRecentTrends(windowSize: number = 100): CausalMap {
    const recent = this.history.slice(-windowSize);
    
    if (recent.length === 0) {
      return {};
    }

    const groups: Record<string, number[]> = {};

    for (const record of recent) {
      const tags = record.coherence.semanticTags;
      
      for (const tag of tags) {
        if (!groups[tag]) {
          groups[tag] = [];
        }
        groups[tag].push(record.coherence.relevance);
      }
    }

    const causalMap: CausalMap = {};

    for (const tag of Object.keys(groups)) {
      const relevances = groups[tag];
      const avg = relevances.reduce((a, b) => a + b, 0) / relevances.length;
      causalMap[tag] = avg;
    }

    return causalMap;
  }

  /**
   * Get statistics about causal relationships
   */
  getStats() {
    return {
      totalRecords: this.history.length,
      uniqueEffects: new Set(this.history.map(r => r.effect)).size,
      uniqueTags: new Set(
        this.history.flatMap(r => r.coherence.semanticTags)
      ).size,
      averageRelevance: this.history.length > 0
        ? this.history.reduce((sum, r) => sum + r.coherence.relevance, 0) / this.history.length
        : 0
    };
  }
}

export const NeuralCausality = new NeuralCausalityEngine();

