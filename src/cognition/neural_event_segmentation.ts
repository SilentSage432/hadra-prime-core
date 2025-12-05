// A115 — Neural Event Segmentation Engine
// Detects boundaries in neural signals to create event segments
// PRIME's brain learns to see events inside its own neural stream

import type { NeuralSignal, CoherenceResult } from "./neural_symbolic_coherence.ts";

export interface NeuralEvent {
  id: string;
  signals: NeuralSignal[];
  coherenceStats: {
    avgRelevance: number;
    avgCoherence: number;
    variance: number;
  };
  start: number;
  end: number;
  boundaryReason?: string; // Why this event ended
}

export class NeuralEventSegmentationEngine {
  private currentEvent: NeuralEvent | null = null;
  private events: NeuralEvent[] = [];
  private maxEvents = 1000; // Keep bounded history

  private threshold = {
    relevanceJump: 0.15,      // Significant relevance change
    coherenceShift: 0.20,     // Significant coherence change
    timeGap: 8000,            // 8 seconds between signals = new event
    varianceSpike: 0.25       // Variance spike indicates boundary
  };

  /**
   * Start a new neural event segment
   */
  startNewEvent(initial: NeuralSignal, coherence: CoherenceResult): NeuralEvent {
    this.currentEvent = {
      id: `evt_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      signals: [initial],
      coherenceStats: {
        avgRelevance: coherence.relevance,
        avgCoherence: coherence.confidence, // Using confidence as coherence proxy
        variance: 0
      },
      start: Date.now(),
      end: Date.now(),
    };

    console.log("[PRIME-EVENT] New event started:", this.currentEvent.id);
    return this.currentEvent;
  }

  /**
   * Add a signal to the current event or start a new one if boundary detected
   */
  addSignal(signal: NeuralSignal, coherence: CoherenceResult): void {
    if (!this.currentEvent) {
      this.startNewEvent(signal, coherence);
      return;
    }

    const lastCoherence = this.currentEvent.signals.length > 0
      ? this.getLastCoherence()
      : { relevance: this.currentEvent.coherenceStats.avgRelevance, confidence: this.currentEvent.coherenceStats.avgCoherence };

    // Compute shifts
    const relevanceShift = Math.abs(coherence.relevance - this.currentEvent.coherenceStats.avgRelevance);
    const coherenceShift = Math.abs(coherence.confidence - this.currentEvent.coherenceStats.avgCoherence);
    const timeGap = Date.now() - this.currentEvent.end;

    // Compute variance of recent signals
    const recentRelevances = this.currentEvent.signals
      .slice(-10)
      .map(s => this.getSignalRelevance(s))
      .filter(r => r !== null) as number[];
    
    const currentVariance = this.computeVariance([
      ...recentRelevances,
      coherence.relevance
    ]);
    const previousVariance = this.computeVariance(recentRelevances);
    const varianceSpike = currentVariance - previousVariance;

    // Detect boundary conditions
    let boundary = false;
    let boundaryReason = "";

    if (relevanceShift > this.threshold.relevanceJump) {
      boundary = true;
      boundaryReason = `relevance shift (${relevanceShift.toFixed(3)})`;
    } else if (coherenceShift > this.threshold.coherenceShift) {
      boundary = true;
      boundaryReason = `coherence shift (${coherenceShift.toFixed(3)})`;
    } else if (timeGap > this.threshold.timeGap) {
      boundary = true;
      boundaryReason = `time gap (${timeGap}ms)`;
    } else if (varianceSpike > this.threshold.varianceSpike) {
      boundary = true;
      boundaryReason = `variance spike (${varianceSpike.toFixed(3)})`;
    }

    if (boundary) {
      console.log(`[PRIME-EVENT] Boundary detected: ${boundaryReason}`);
      if (relevanceShift > this.threshold.relevanceJump) {
        console.log(`[PRIME-COHERENCE] relevance shift=${relevanceShift.toFixed(2)} → event boundary`);
      }
      this.finalizeEvent(boundaryReason);
      this.startNewEvent(signal, coherence);
    } else {
      // Add signal to current event
      this.currentEvent.signals.push(signal);
      this.currentEvent.end = Date.now();

      // Update running averages
      const totalSignals = this.currentEvent.signals.length;
      this.currentEvent.coherenceStats.avgRelevance = 
        (this.currentEvent.coherenceStats.avgRelevance * (totalSignals - 1) + coherence.relevance) / totalSignals;
      this.currentEvent.coherenceStats.avgCoherence = 
        (this.currentEvent.coherenceStats.avgCoherence * (totalSignals - 1) + coherence.confidence) / totalSignals;
      
      // Update variance
      const allRelevances = this.currentEvent.signals.map(s => this.getSignalRelevance(s)).filter(r => r !== null) as number[];
      this.currentEvent.coherenceStats.variance = this.computeVariance(allRelevances);
    }
  }

  /**
   * Finalize the current event and add it to history
   */
  finalizeEvent(reason?: string): NeuralEvent | null {
    if (!this.currentEvent) return null;

    if (reason) {
      this.currentEvent.boundaryReason = reason;
    }

    this.events.push(this.currentEvent);
    
    // Maintain bounded history
    if (this.events.length > this.maxEvents) {
      this.events.shift();
    }

    const finalized = this.currentEvent;
    this.currentEvent = null;

    console.log(`[PRIME-EVENT] Event finalized: ${finalized.id} (${finalized.signals.length} signals, duration: ${finalized.end - finalized.start}ms)`);
    
    return finalized;
  }

  /**
   * Get recent events
   */
  getRecentEvents(count: number = 5): NeuralEvent[] {
    return this.events.slice(-count);
  }

  /**
   * Get current active event
   */
  getCurrentEvent(): NeuralEvent | null {
    return this.currentEvent;
  }

  /**
   * Get all events
   */
  getAllEvents(): NeuralEvent[] {
    return [...this.events];
  }

  /**
   * Helper: Get relevance from a signal (if we have coherence data cached)
   */
  private getSignalRelevance(signal: NeuralSignal): number | null {
    // For now, compute magnitude as proxy
    if (signal.vector && signal.vector.length > 0) {
      return signal.vector.reduce((a, b) => a + Math.abs(b), 0) / signal.vector.length;
    }
    return null;
  }

  /**
   * Helper: Get last coherence stats (simplified)
   */
  private getLastCoherence(): { relevance: number; confidence: number } {
    return {
      relevance: this.currentEvent?.coherenceStats.avgRelevance || 0.5,
      confidence: this.currentEvent?.coherenceStats.avgCoherence || 0.5
    };
  }

  /**
   * Compute variance of values
   */
  private computeVariance(values: number[]): number {
    if (values.length === 0) return 0;
    
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const squaredDiffs = values.map(v => Math.pow(v - mean, 2));
    const variance = squaredDiffs.reduce((a, b) => a + b, 0) / values.length;
    
    return variance;
  }
}

export const NeuralEventSegmentation = new NeuralEventSegmentationEngine();

