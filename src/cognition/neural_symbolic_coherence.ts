// A113 — Neural-Symbolic Coherence Engine
// Interprets neural vectors as symbolic meaning
// Bridges neural outputs with symbolic reasoning

export interface NeuralSignal {
  vector: number[];
  slot: number;
  timestamp: number;
  source?: string; // e.g., "temporal_embedding", "context_encoding"
}

export interface CoherenceResult {
  semanticTags: string[];
  relevance: number;
  confidence: number;
  magnitude: number;
  contextAnchors?: string[];
}

export class NeuralSymbolicCoherenceEngine {
  /**
   * Interpret a neural signal and convert it to symbolic meaning
   */
  interpret(signal: NeuralSignal): CoherenceResult {
    if (!signal.vector || signal.vector.length === 0) {
      return {
        semanticTags: ["empty_signal"],
        relevance: 0,
        confidence: 0,
        magnitude: 0
      };
    }

    // Compute magnitude (average absolute value)
    const magnitude =
      signal.vector.reduce((a, b) => a + Math.abs(b), 0) / signal.vector.length;

    // Generate semantic tags based on magnitude and vector characteristics
    const tags: string[] = [];
    
    if (magnitude > 0.6) {
      tags.push("high_relevance");
    } else if (magnitude > 0.3) {
      tags.push("moderate_relevance");
    } else {
      tags.push("low_relevance");
    }

    // Analyze vector variance (high variance = more complex signal)
    const variance = this.computeVariance(signal.vector);
    if (variance > 0.1) {
      tags.push("high_variance");
    } else {
      tags.push("low_variance");
    }

    // Check for temporal patterns (if vector length suggests sequence)
    if (signal.vector.length >= 32) {
      tags.push("temporal_sequence");
    }

    // Source-specific tags
    if (signal.source) {
      tags.push(`source_${signal.source}`);
    }

    // Compute relevance (normalized magnitude)
    const relevance = Math.min(1, magnitude);

    // Compute confidence (magnitude with slight boost, capped at 1)
    const confidence = Math.min(1, magnitude * 1.2);

    // Generate context anchors based on vector characteristics
    const contextAnchors = this.generateContextAnchors(signal.vector, magnitude);

    return {
      semanticTags: tags,
      relevance,
      confidence,
      magnitude,
      contextAnchors
    };
  }

  /**
   * Compute variance of vector values
   */
  private computeVariance(vector: number[]): number {
    if (vector.length === 0) return 0;
    
    const mean = vector.reduce((a, b) => a + b, 0) / vector.length;
    const squaredDiffs = vector.map(v => Math.pow(v - mean, 2));
    const variance = squaredDiffs.reduce((a, b) => a + b, 0) / vector.length;
    
    return variance;
  }

  /**
   * Generate context anchors from vector characteristics
   */
  private generateContextAnchors(vector: number[], magnitude: number): string[] {
    const anchors: string[] = [];

    // High magnitude suggests strong context
    if (magnitude > 0.5) {
      anchors.push("strong_context");
    }

    // Check for dominant values (peaks in the vector)
    const maxVal = Math.max(...vector.map(Math.abs));
    if (maxVal > 0.7) {
      anchors.push("dominant_features");
    }

    // Check for sparse vs dense vectors
    const nonZeroCount = vector.filter(v => Math.abs(v) > 0.01).length;
    const sparsity = nonZeroCount / vector.length;
    if (sparsity < 0.3) {
      anchors.push("sparse_encoding");
    } else {
      anchors.push("dense_encoding");
    }

    return anchors;
  }

  /**
   * Evaluate coherence between multiple neural signals
   */
  evaluateCoherence(signals: NeuralSignal[]): {
    overallRelevance: number;
    coherenceScore: number;
    dominantTags: string[];
  } {
    if (signals.length === 0) {
      return {
        overallRelevance: 0,
        coherenceScore: 0,
        dominantTags: []
      };
    }

    const interpretations = signals.map(s => this.interpret(s));
    const avgRelevance = interpretations.reduce((sum, i) => sum + i.relevance, 0) / interpretations.length;
    const avgConfidence = interpretations.reduce((sum, i) => sum + i.confidence, 0) / interpretations.length;

    // Collect all tags and find dominant ones
    const tagCounts: Record<string, number> = {};
    interpretations.forEach(i => {
      i.semanticTags.forEach(tag => {
        tagCounts[tag] = (tagCounts[tag] || 0) + 1;
      });
    });

    const dominantTags = Object.entries(tagCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([tag]) => tag);

    // Coherence score: how well signals align
    const coherenceScore = avgConfidence * (1 - this.computeSignalVariance(interpretations));

    return {
      overallRelevance: avgRelevance,
      coherenceScore: Math.max(0, Math.min(1, coherenceScore)),
      dominantTags
    };
  }

  /**
   * Compute variance across interpretations (lower = more coherent)
   */
  private computeSignalVariance(interpretations: CoherenceResult[]): number {
    if (interpretations.length <= 1) return 0;

    const relevances = interpretations.map(i => i.relevance);
    const mean = relevances.reduce((a, b) => a + b, 0) / relevances.length;
    const variance = relevances.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / relevances.length;
    
    return Math.sqrt(variance); // Standard deviation
  }
}

export const NeuralSymbolicCoherence = new NeuralSymbolicCoherenceEngine();

