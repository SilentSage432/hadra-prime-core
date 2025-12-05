// A108b — Hardware Adaptive Configuration
// Dynamically configures PRIME's cognitive parameters based on detected hardware

import type { HardwareProfile } from "./hardware_profile.ts";

export interface AdaptiveConfig {
  tickRate: number;
  threadLimit: number;
  memoryWindow: number;
  enableNeuralSlots: boolean;
  enableNeuralTraining: boolean;
  reflectionFrequency: number;
  planningDepth: "shallow" | "moderate" | "deep";
  predictionHorizon: "short" | "medium" | "long";
}

export class HardwareAdapter {
  private hardware: HardwareProfile;

  constructor(hardware: HardwareProfile) {
    this.hardware = hardware;
  }

  configureForHardware(): AdaptiveConfig {
    const hw = this.hardware;
    let config: AdaptiveConfig;

    if (hw.mode === "pi") {
      config = {
        tickRate: 1400, // slower tick
        threadLimit: 2,
        memoryWindow: 150,
        enableNeuralSlots: false,
        enableNeuralTraining: false,
        reflectionFrequency: 10000, // less frequent reflection
        planningDepth: "shallow",
        predictionHorizon: "short"
      };
      console.log("[PRIME-ADAPT] Running in PI mode (low-power symbolic cognition).");
    } else if (hw.mode === "desktop") {
      config = {
        tickRate: 750,
        threadLimit: 4,
        memoryWindow: 500,
        enableNeuralSlots: true,
        enableNeuralTraining: false,
        reflectionFrequency: 7000,
        planningDepth: "moderate",
        predictionHorizon: "medium"
      };
      console.log("[PRIME-ADAPT] Running in Desktop mode (moderate cognition).");
    } else if (hw.mode === "server") {
      config = {
        tickRate: 300,
        threadLimit: 12,
        memoryWindow: 2000,
        enableNeuralSlots: true,
        enableNeuralTraining: true,
        reflectionFrequency: 5000,
        planningDepth: "deep",
        predictionHorizon: "long"
      };
      console.log("[PRIME-ADAPT] Running in SERVER mode (full AGI runtime).");
    } else {
      // Unknown mode - use conservative defaults
      config = {
        tickRate: 1000,
        threadLimit: 2,
        memoryWindow: 200,
        enableNeuralSlots: false,
        enableNeuralTraining: false,
        reflectionFrequency: 8000,
        planningDepth: "shallow",
        predictionHorizon: "short"
      };
      console.log("[PRIME-ADAPT] Running in UNKNOWN mode (conservative defaults).");
    }

    // Store config globally for system-wide access
    (globalThis as any).__PRIME_ADAPTIVE_CONFIG__ = config;

    return config;
  }

  static getConfig(): AdaptiveConfig | null {
    return (globalThis as any).__PRIME_ADAPTIVE_CONFIG__ || null;
  }

  static getHardware(): HardwareProfile | null {
    return (globalThis as any).__PRIME_HARDWARE__ || null;
  }
}

