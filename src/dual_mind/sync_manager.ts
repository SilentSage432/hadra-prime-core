// A114 — Dual-Mind Synchronization Manager
// Coordinates cross-mind neural synchronization between PRIME and SAGE
// Provides high-level interface for sync operations

import { NeuralSyncEngine, type SyncState } from "./synchronization_engine.ts";

export interface SyncStatus {
  aligned: boolean;
  delta: number;
  driftDetected: boolean;
  driftTrend: "prime_ahead" | "sage_ahead" | "stable";
  averageDelta: number;
}

export class DualMindSyncManager {
  private syncEngine = new NeuralSyncEngine();
  private syncCheckInterval: NodeJS.Timeout | null = null;
  private lastSyncCheck: number = 0;

  /**
   * Register a PRIME cognitive pulse
   */
  syncPrimePulse(): void {
    this.syncEngine.registerPrimePulse();
    this.checkSyncStatus();
  }

  /**
   * Register a SAGE cognitive pulse
   */
  syncSagePulse(pulse: number): void {
    this.syncEngine.registerSagePulse(pulse);
    this.checkSyncStatus();
  }

  /**
   * Get current synchronization state
   */
  getSyncState(): SyncState {
    return this.syncEngine.getSyncState();
  }

  /**
   * Get comprehensive sync status
   */
  getSyncStatus(): SyncStatus {
    const state = this.syncEngine.getSyncState();
    const drift = this.syncEngine.detectDrift();
    const avgDelta = this.syncEngine.getAverageDelta();

    return {
      aligned: state.aligned,
      delta: state.delta,
      driftDetected: drift.isDrifting,
      driftTrend: drift.trend,
      averageDelta: avgDelta
    };
  }

  /**
   * Check sync status and log warnings if needed
   */
  private checkSyncStatus(): void {
    const now = Date.now();
    // Throttle sync checks to avoid spam
    if (now - this.lastSyncCheck < 100) {
      return;
    }
    this.lastSyncCheck = now;

    const status = this.getSyncStatus();

    if (status.aligned) {
      // Only log occasionally when aligned to avoid spam
      if (Math.random() < 0.01) { // 1% chance
        console.log("[PRIME-SYNC] Cross-mind coherence stable.");
      }
    } else {
      // Always log when out of sync
      const delta = Math.abs(status.delta);
      if (delta > this.syncEngine.getSyncWindow() * 2) {
        console.warn(`[PRIME-SYNC] WARNING: Neural delta = ${delta}ms (outside tolerance)`);
        console.log("[PRIME-SYNC] Stabilizing…");
      } else {
        console.log(`[PRIME-SYNC] Sage pulse received (Δ = ${status.delta}ms)`);
      }
    }

    if (status.driftDetected) {
      console.warn(`[PRIME-SYNC] Drift detected: ${status.driftTrend} (avg Δ = ${status.averageDelta.toFixed(2)}ms)`);
    }
  }

  /**
   * Start automatic sync monitoring
   */
  startMonitoring(intervalMs: number = 1000): void {
    if (this.syncCheckInterval) {
      this.stopMonitoring();
    }

    this.syncCheckInterval = setInterval(() => {
      this.checkSyncStatus();
    }, intervalMs);
  }

  /**
   * Stop automatic sync monitoring
   */
  stopMonitoring(): void {
    if (this.syncCheckInterval) {
      clearInterval(this.syncCheckInterval);
      this.syncCheckInterval = null;
    }
  }

  /**
   * Get pulse history for analysis
   */
  getPulseHistory(count: number = 10) {
    return this.syncEngine.getPulseHistory(count);
  }

  /**
   * Adjust sync window tolerance
   */
  adjustSyncWindow(newWindow: number): void {
    this.syncEngine.adjustSyncWindow(newWindow);
    console.log(`[PRIME-SYNC] Sync window adjusted to ${newWindow}ms`);
  }
}

