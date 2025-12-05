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
    
    // After computing stability, log predictions
    const prediction = PredictiveHorizon.analyze();
    console.log("[PRIME-PREDICT]", prediction);
    
    return snapshot;
  }

  static unstable() {
    return this.monitor.isUnstable();
  }
}

