// A115 — Neural Event Segmentation Engine
// Detects where one cognitive event ends and another begins
// Enables structured experience, memory segmentation, and neural markers for shifts
// This is precursor-level consciousness engineering

import { primeEmbeddingAdapter } from "./embedding/prime_embedding_adapter.ts";
import { PRIME_TEMPORAL } from "../temporal/reasoner.ts";
import type { EventBoundary } from "../memory/episodic/event_capture.ts";

export interface CognitiveState {
  intent?: string;
  emotion?: any;
  predictionError?: number;
  embedding?: number[];
  goal?: string;
  stability?: number;
  [key: string]: any;
}

export class EventSegmentationEngine {
  private lastEmbedding: number[] | null = null;
  private lastIntent: string | null = null;
  private lastEmotion: any = null;
  private lastGoal: string | null = null;
  private threshold = 0.22; // Adaptive threshold for embedding delta
  private boundaries: EventBoundary[] = [];
  private maxBoundaries = 1000;

  /**
   * Main entry: detect whether a new cognitive event just occurred
   */
  detectEventBoundary(state: CognitiveState): EventBoundary | null {
    // Extract or generate embedding
    let embedding: number[];
    if (state.embedding && Array.isArray(state.embedding)) {
      embedding = state.embedding;
    } else {
      // Generate embedding from state
      embedding = this.generateEmbeddingFromState(state);
    }

    // Compute embedding delta
    const delta = this.computeDelta(embedding);

    // Check for intent change
    const intentChanged = state.intent && state.intent !== this.lastIntent;

    // Check for emotion shift
    const emotionShift = this.emotionDelta(state.emotion);

    // Check for goal change
    const goalChanged = state.goal && state.goal !== this.lastGoal;

    // Check for prediction error spike
    const predictionError = state.predictionError || 0;

    // Determine if we should segment
    const shouldSegment =
      delta > this.threshold ||
      intentChanged ||
      emotionShift > 0.18 ||
      predictionError > 0.25 ||
      goalChanged;

    // Update tracking state
    this.lastEmbedding = embedding;
    if (state.intent) this.lastIntent = state.intent;
    if (state.emotion) this.lastEmotion = state.emotion;
    if (state.goal) this.lastGoal = state.goal;

    if (!shouldSegment) return null;

    // Create event boundary
    const boundary: EventBoundary = {
      id: this.generateId(),
      timestamp: Date.now(),
      embedding: embedding,
      intent: state.intent || "unknown",
      emotion: state.emotion || {},
      predictionError: predictionError,
      windowRef: this.getTemporalWindowSnapshot(),
      goal: state.goal || null,
      delta: delta,
      emotionShift: emotionShift,
      reason: this.determineBoundaryReason(delta, intentChanged, emotionShift, predictionError, goalChanged)
    };

    // Store boundary
    this.boundaries.push(boundary);
    if (this.boundaries.length > this.maxBoundaries) {
      this.boundaries.shift();
    }

    console.log("[PRIME-EVENT-SEG] Event boundary detected:", {
      id: boundary.id,
      reason: boundary.reason,
      delta: delta.toFixed(3),
      intent: boundary.intent
    });

    return boundary;
  }

  /**
   * Compute embedding delta (Euclidean distance)
   */
  private computeDelta(vec: number[]): number {
    if (!this.lastEmbedding || this.lastEmbedding.length === 0) {
      return 1.0; // First embedding = maximum delta
    }

    if (vec.length !== this.lastEmbedding.length) {
      // Different dimensions, normalize or pad
      const minLen = Math.min(vec.length, this.lastEmbedding.length);
      let sum = 0;
      for (let i = 0; i < minLen; i++) {
        const diff = vec[i] - this.lastEmbedding[i];
        sum += diff * diff;
      }
      return Math.sqrt(sum);
    }

    let sum = 0;
    for (let i = 0; i < vec.length; i++) {
      const diff = vec[i] - this.lastEmbedding[i];
      sum += diff * diff;
    }
    return Math.sqrt(sum);
  }

  /**
   * Compute emotion delta
   */
  private emotionDelta(newEmotion: any): number {
    if (!newEmotion || typeof newEmotion !== "object") return 0;
    if (!this.lastEmotion || typeof this.lastEmotion !== "object") return 0.5;

    let diff = 0;
    let count = 0;

    // Check all emotion keys
    const allKeys = new Set([...Object.keys(newEmotion), ...Object.keys(this.lastEmotion)]);
    
    for (const key of allKeys) {
      const newVal = newEmotion[key] || 0;
      const oldVal = this.lastEmotion[key] || 0;
      diff += Math.abs(newVal - oldVal);
      count++;
    }

    return count > 0 ? diff / count : 0;
  }

  /**
   * Generate embedding from cognitive state
   */
  private generateEmbeddingFromState(state: CognitiveState): number[] {
    const features: number[] = [];

    // Intent encoding
    if (state.intent) {
      const intentHash = this.hashString(state.intent);
      features.push(intentHash);
    }

    // Goal encoding
    if (state.goal) {
      const goalHash = this.hashString(state.goal);
      features.push(goalHash);
    }

    // Emotion encoding
    if (state.emotion && typeof state.emotion === "object") {
      const emotionKeys = Object.keys(state.emotion);
      emotionKeys.forEach(key => {
        features.push(state.emotion[key] || 0);
      });
    }

    // Stability encoding
    if (state.stability !== undefined) {
      features.push(state.stability);
    }

    // Prediction error encoding
    if (state.predictionError !== undefined) {
      features.push(state.predictionError);
    }

    // Normalize to 64 dimensions (standard PRIME embedding size)
    while (features.length < 64) {
      features.push(0);
    }
    return features.slice(0, 64);
  }

  /**
   * Hash string to number
   */
  private hashString(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash) / 2147483647; // Normalize to 0-1
  }

  /**
   * Get temporal window snapshot
   */
  private getTemporalWindowSnapshot(): any {
    try {
      // Get recent temporal snapshots
      const shortWindow = PRIME_TEMPORAL.short.getAll();
      const mediumWindow = PRIME_TEMPORAL.medium.getAll();
      
      return {
        short: shortWindow.slice(-10), // Last 10 short-term snapshots
        medium: mediumWindow.slice(-5), // Last 5 medium-term snapshots
        timestamp: Date.now()
      };
    } catch (error) {
      console.warn("[PRIME-EVENT-SEG] Error getting temporal window snapshot:", error);
      return { timestamp: Date.now() };
    }
  }

  /**
   * Determine why this boundary was detected
   */
  private determineBoundaryReason(
    delta: number,
    intentChanged: boolean,
    emotionShift: number,
    predictionError: number,
    goalChanged: boolean
  ): string {
    const reasons: string[] = [];

    if (delta > this.threshold) {
      reasons.push(`embedding_delta(${delta.toFixed(3)})`);
    }
    if (intentChanged) {
      reasons.push("intent_shift");
    }
    if (emotionShift > 0.18) {
      reasons.push(`emotion_shift(${emotionShift.toFixed(3)})`);
    }
    if (predictionError > 0.25) {
      reasons.push(`prediction_error(${predictionError.toFixed(3)})`);
    }
    if (goalChanged) {
      reasons.push("goal_change");
    }

    return reasons.join(" + ") || "unknown";
  }

  /**
   * Generate unique ID for event boundary
   */
  private generateId(): string {
    return `evt_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Get recent boundaries
   */
  getRecentBoundaries(count: number = 10): EventBoundary[] {
    return this.boundaries.slice(-count);
  }

  /**
   * Adjust threshold dynamically
   */
  adjustThreshold(newThreshold: number): void {
    if (newThreshold > 0 && newThreshold <= 1.0) {
      this.threshold = newThreshold;
      console.log(`[PRIME-EVENT-SEG] Threshold adjusted to ${newThreshold}`);
    }
  }

  /**
   * Get current threshold
   */
  getThreshold(): number {
    return this.threshold;
  }
}

