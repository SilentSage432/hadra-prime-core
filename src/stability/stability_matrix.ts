// src/stability/stability_matrix.ts

import { StabilityMonitor } from "./stability_monitor.ts";

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
    return this.monitor.snapshot();
  }

  static unstable() {
    return this.monitor.isUnstable();
  }
}

