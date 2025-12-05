// src/stability/stability_matrix.ts

import { StabilityMonitor } from "./stability_monitor.ts";
import { PredictiveHorizon } from "../cognition/predictive_horizon.ts";

export class StabilityMatrix {
  static monitor = new StabilityMonitor();

  static init() {
    this.monitor.registerSubsystem("perception");
    this.monitor.registerSubsystem("cognition");
    this.monitor.registerSubsystem("intent_engine");
    this.monitor.registerSubsystem("memory");
    this.monitor.registerSubsystem("expression");
    this.monitor.registerSubsystem("meta");

    console.log("[PRIME] Stability Matrix initialized.");
  }

  static update(subsystem: string, metrics: any) {
    this.monitor.updateSubsystem(subsystem, metrics);
  }

  static getSnapshot() {
    const snapshot = this.monitor.snapshot();
    
    // FIXED: Removed automatic prediction trigger to prevent recursion storms
    // Predictions should only be generated when explicitly requested via events
    // const prediction = PredictiveHorizon.analyze();
    // console.log("[PRIME-PREDICT]", prediction);
    
    return snapshot;
  }

  static unstable() {
    return this.monitor.isUnstable();
  }

  // A41: Reinforce internal invariants when protection mode is needed
  static reinforceInvariant() {
    // Strengthen stability monitoring when protection is needed
    // This is a lightweight internal action that doesn't modify external state
    const snapshot = this.getSnapshot();
    if (snapshot && snapshot.score < 0.5) {
      // Internal flag for protection mode - no external side effects
      console.log("[PRIME-STABILITY] Protection mode reinforced");
    }
  }
}

