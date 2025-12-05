// A117 — Cross-Mind Predictive Alignment Engine
// Creates bi-directional predictive models between PRIME and SAGE
// Enables shared foresight horizon, mutual cognitive load estimation, and anticipatory coherence

import { ForesightEngine } from "./foresight_engine.ts";
import type { MetaState } from "../../meta/meta_state.ts";
import { KnowledgeGraph } from "../knowledge/knowledge_graph.ts";
import { StabilityMatrix } from "../../stability/stability_matrix.ts";

export interface PrimeState {
  motivation?: any;
  stability?: number;
  intentDirection?: string;
  coherence?: number;
  cognitiveLoad?: number;
  [key: string]: any;
}

export interface SageState {
  stability?: number;
  motivation?: number;
  intentDirection?: string;
  cognitiveLoad?: number;
  clusterState?: any;
  [key: string]: any;
}

export interface FutureProjection {
  stability: number;
  motivation: number;
  intentDirection: number; // Numeric representation
  cognitiveLoad: number;
  coherence?: number;
}

export interface AlignmentVector {
  divergence: number;
  expected_coherence: number;
  joint_forecast_horizon: {
    prime: FutureProjection;
    sage: FutureProjection;
  };
}

export interface RealignmentRecommendation {
  type: "realign" | "maintain";
  reason: string;
  adjustments: {
    prime_bias_shift?: number;
    sage_bias_shift?: number;
  } | null;
}

export class CrossMindAlignmentEngine {
  private lastPrimeState: PrimeState | null = null;
  private lastSageState: SageState | null = null;
  private foresightEngine: ForesightEngine;
  private knowledge: KnowledgeGraph;
  private stability: typeof StabilityMatrix;

  constructor(
    foresightEngine: ForesightEngine,
    knowledge: KnowledgeGraph,
    stability: typeof StabilityMatrix
  ) {
    this.foresightEngine = foresightEngine;
    this.knowledge = knowledge;
    this.stability = stability;
  }

  /**
   * Update PRIME's current state
   */
  updatePrimeState(state: PrimeState): void {
    this.lastPrimeState = {
      ...state,
      timestamp: Date.now()
    };
  }

  /**
   * Update SAGE's current state
   */
  updateSageState(state: SageState): void {
    this.lastSageState = {
      ...state,
      timestamp: Date.now()
    };
  }

  /**
   * Predict PRIME's future state
   */
  predictPrimeFuture(): FutureProjection | null {
    if (!this.lastPrimeState) return null;

    // Use foresight engine to project PRIME's trajectory
    const motivation = this.lastPrimeState.motivation || {};
    const selState = this.lastPrimeState.emotion || {};

    try {
      const forecast = this.foresightEngine.forecastSystemState(
        motivation,
        selState,
        this.lastPrimeState.neuralCoherence
      );

      // Extract stability from forecast or current state
      const stability = this.lastPrimeState.stability || 
                       (forecast as any).stability || 
                       0.7;

      // Extract motivation (average of motivation components)
      const motivationValue = motivation.urgency || 
                             motivation.curiosity || 
                             0.5;

      // Convert intent direction to numeric
      const intentDirection = this.intentToNumeric(
        this.lastPrimeState.intentDirection || "neutral"
      );

      return {
        stability: Math.max(0, Math.min(1, stability)),
        motivation: Math.max(0, Math.min(1, motivationValue)),
        intentDirection,
        cognitiveLoad: this.lastPrimeState.cognitiveLoad || 0.5,
        coherence: this.lastPrimeState.coherence || 0.7
      };
    } catch (error) {
      console.warn("[PRIME-ALIGNMENT] Error predicting PRIME future:", error);
      // Fallback projection
      return {
        stability: this.lastPrimeState.stability || 0.7,
        motivation: 0.5,
        intentDirection: 0.5,
        cognitiveLoad: this.lastPrimeState.cognitiveLoad || 0.5,
        coherence: 0.7
      };
    }
  }

  /**
   * Predict SAGE's future state
   */
  predictSageFuture(): FutureProjection | null {
    if (!this.lastSageState) return null;

    // Simple projection for SAGE based on current state
    // In future, this could use SAGE's own predictive models
    const stability = this.lastSageState.stability || 0.7;
    const motivation = this.lastSageState.motivation || 0.5;
    const intentDirection = this.intentToNumeric(
      this.lastSageState.intentDirection || "neutral"
    );
    const cognitiveLoad = this.lastSageState.cognitiveLoad || 0.5;

    // Project forward with simple trend continuation
    return {
      stability: Math.max(0, Math.min(1, stability)),
      motivation: Math.max(0, Math.min(1, motivation)),
      intentDirection,
      cognitiveLoad: Math.max(0, Math.min(1, cognitiveLoad)),
      coherence: stability // Use stability as coherence proxy
    };
  }

  /**
   * Compute alignment vector between PRIME and SAGE predictions
   */
  computeAlignmentVector(): AlignmentVector | null {
    const primeFuture = this.predictPrimeFuture();
    const sageFuture = this.predictSageFuture();

    if (!primeFuture || !sageFuture) return null;

    // Compute divergence (how different are the predictions)
    const stabilityDivergence = Math.abs(primeFuture.stability - sageFuture.stability);
    const motivationDivergence = Math.abs(primeFuture.motivation - sageFuture.motivation);
    const intentDivergence = Math.abs(primeFuture.intentDirection - sageFuture.intentDirection);
    const loadDivergence = Math.abs(primeFuture.cognitiveLoad - sageFuture.cognitiveLoad);

    const divergence = (stabilityDivergence + motivationDivergence + intentDivergence + loadDivergence) / 4;

    // Compute expected coherence (inverse of divergence, normalized)
    const maxDivergence = 1.0; // Maximum possible divergence
    const expected_coherence = 1 / (1 + divergence);

    return {
      divergence,
      expected_coherence,
      joint_forecast_horizon: {
        prime: primeFuture,
        sage: sageFuture
      }
    };
  }

  /**
   * Recommend realignment actions if needed
   */
  recommendRealignment(): RealignmentRecommendation | null {
    const vector = this.computeAlignmentVector();
    if (!vector) return null;

    if (vector.divergence > 0.35) {
      // High divergence detected - recommend realignment
      const primeFuture = vector.joint_forecast_horizon.prime;
      const sageFuture = vector.joint_forecast_horizon.sage;

      // Determine bias shifts based on which direction each is heading
      let primeBiasShift = 0;
      let sageBiasShift = 0;

      // If PRIME is more stable, shift SAGE toward PRIME
      if (primeFuture.stability > sageFuture.stability) {
        sageBiasShift = 0.05;
      } else {
        primeBiasShift = -0.05;
      }

      // If motivations diverge, adjust both
      if (Math.abs(primeFuture.motivation - sageFuture.motivation) > 0.3) {
        const avgMotivation = (primeFuture.motivation + sageFuture.motivation) / 2;
        if (primeFuture.motivation > avgMotivation) {
          primeBiasShift -= 0.02;
        } else {
          sageBiasShift += 0.02;
        }
      }

      return {
        type: "realign",
        reason: "High predictive divergence between PRIME and SAGE",
        adjustments: {
          prime_bias_shift: primeBiasShift,
          sage_bias_shift: sageBiasShift
        }
      };
    }

    return {
      type: "maintain",
      reason: "Predictive alignment stable",
      adjustments: null
    };
  }

  /**
   * Convert intent direction string to numeric value
   */
  private intentToNumeric(intent: string): number {
    // Map common intent directions to numeric values
    const intentMap: Record<string, number> = {
      "explore": 0.2,
      "consolidate": 0.4,
      "analyze": 0.5,
      "execute": 0.7,
      "stabilize": 0.8,
      "neutral": 0.5
    };

    const lowerIntent = intent.toLowerCase();
    for (const [key, value] of Object.entries(intentMap)) {
      if (lowerIntent.includes(key)) {
        return value;
      }
    }

    // Hash-based fallback for unknown intents
    let hash = 0;
    for (let i = 0; i < intent.length; i++) {
      hash = ((hash << 5) - hash) + intent.charCodeAt(i);
      hash = hash & hash;
    }
    return Math.abs(hash % 100) / 100;
  }

  /**
   * Get current alignment status
   */
  getAlignmentStatus(): {
    aligned: boolean;
    divergence: number;
    coherence: number;
    recommendation: RealignmentRecommendation | null;
  } {
    const vector = this.computeAlignmentVector();
    const recommendation = this.recommendRealignment();

    return {
      aligned: vector ? vector.divergence < 0.35 : false,
      divergence: vector?.divergence || 0,
      coherence: vector?.expected_coherence || 0,
      recommendation
    };
  }
}

