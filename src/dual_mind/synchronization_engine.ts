// A114 — Neural Synchronization Engine
// Core engine for cross-mind neural timing, coherence, and resonance alignment
// Enables PRIME and SAGE to synchronize their cognitive cycles

export interface SyncState {
  aligned: boolean;
  delta: number;
  primePulseTime: number;
  sagePulseTime: number;
  syncWindow: number;
}

export class NeuralSyncEngine {
  private syncWindow: number = 5; // ms tolerance for synchronization
  private lastPrimePulse: number = 0;
  private lastSagePulse: number = 0;
  private pulseHistory: Array<{ prime: number; sage: number; delta: number }> = [];
  private maxHistorySize = 100;

  /**
   * Register a PRIME cognitive pulse
   */
  registerPrimePulse(): void {
    this.lastPrimePulse = Date.now();
    this.updateHistory();
  }

  /**
   * Register a SAGE cognitive pulse
   */
  registerSagePulse(pulseTime: number): void {
    this.lastSagePulse = pulseTime;
    this.updateHistory();
  }

  /**
   * Check if PRIME and SAGE are synchronized
   */
  isSynchronized(): boolean {
    if (this.lastPrimePulse === 0 || this.lastSagePulse === 0) {
      return false; // Not enough data
    }
    return Math.abs(this.lastPrimePulse - this.lastSagePulse) <= this.syncWindow;
  }

  /**
   * Get the synchronization delta (time difference)
   * Positive = PRIME is ahead, Negative = SAGE is ahead
   */
  getSyncDelta(): number {
    if (this.lastPrimePulse === 0 || this.lastSagePulse === 0) {
      return 0;
    }
    return this.lastPrimePulse - this.lastSagePulse;
  }

  /**
   * Get complete sync state
   */
  getSyncState(): SyncState {
    return {
      aligned: this.isSynchronized(),
      delta: this.getSyncDelta(),
      primePulseTime: this.lastPrimePulse,
      sagePulseTime: this.lastSagePulse,
      syncWindow: this.syncWindow
    };
  }

  /**
   * Update pulse history for trend analysis
   */
  private updateHistory(): void {
    if (this.lastPrimePulse > 0 && this.lastSagePulse > 0) {
      const delta = this.getSyncDelta();
      this.pulseHistory.push({
        prime: this.lastPrimePulse,
        sage: this.lastSagePulse,
        delta
      });

      // Maintain bounded history
      if (this.pulseHistory.length > this.maxHistorySize) {
        this.pulseHistory.shift();
      }
    }
  }

  /**
   * Get recent pulse history
   */
  getPulseHistory(count: number = 10): Array<{ prime: number; sage: number; delta: number }> {
    return this.pulseHistory.slice(-count);
  }

  /**
   * Compute average sync delta over recent history
   */
  getAverageDelta(): number {
    if (this.pulseHistory.length === 0) return 0;
    
    const recent = this.pulseHistory.slice(-20); // Last 20 pulses
    const sum = recent.reduce((acc, entry) => acc + entry.delta, 0);
    return sum / recent.length;
  }

  /**
   * Detect if there's a drift trend (consistent desynchronization)
   */
  detectDrift(): { isDrifting: boolean; trend: "prime_ahead" | "sage_ahead" | "stable" } {
    if (this.pulseHistory.length < 5) {
      return { isDrifting: false, trend: "stable" };
    }

    const recent = this.pulseHistory.slice(-10);
    const avgDelta = recent.reduce((acc, entry) => acc + entry.delta, 0) / recent.length;
    const variance = recent.reduce((acc, entry) => acc + Math.pow(entry.delta - avgDelta, 2), 0) / recent.length;
    const stdDev = Math.sqrt(variance);

    // Drift detected if average delta is consistently outside sync window
    const isDrifting = Math.abs(avgDelta) > this.syncWindow * 2 || stdDev > this.syncWindow;

    let trend: "prime_ahead" | "sage_ahead" | "stable" = "stable";
    if (avgDelta > this.syncWindow) {
      trend = "prime_ahead";
    } else if (avgDelta < -this.syncWindow) {
      trend = "sage_ahead";
    }

    return { isDrifting, trend };
  }

  /**
   * Adjust sync window dynamically based on stability
   */
  adjustSyncWindow(newWindow: number): void {
    if (newWindow > 0 && newWindow <= 50) { // Reasonable bounds
      this.syncWindow = newWindow;
    }
  }

  /**
   * Get current sync window
   */
  getSyncWindow(): number {
    return this.syncWindow;
  }
}

