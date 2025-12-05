// A108b — Hardware Profiling Layer
// PRIME gains self-awareness of her computational substrate
// Enables adaptive cognition, neural offloading, and hardware-optimized behavior

import os from "os";

export interface HardwareProfile {
  mode: "pi" | "server" | "cluster" | "desktop" | "unknown";
  cpuCount: number;
  totalMem: number;
  freeMem: number;
  platform: string;
  arch: string;
  thermalMode?: "normal" | "hot" | "critical";
  neuralCapacity: "none" | "tiny" | "small" | "medium" | "full";
}

export class HardwareProfiler {
  static detect(): HardwareProfile {
    const cpuCount = os.cpus()?.length || 1;
    const totalMem = os.totalmem();
    const freeMem = os.freemem();
    const platform = os.platform();
    const arch = os.arch();

    let mode: HardwareProfile["mode"] = "unknown";

    // Raspberry Pi detection
    if (platform === "linux" && arch === "arm64") {
      mode = "pi";
    }

    // Mac / PC local dev
    if (platform === "darwin") {
      mode = "desktop";
    }

    // Server detection (Talos, Ubuntu, Kubernetes node, etc.)
    if (platform === "linux" && arch !== "arm64") {
      mode = "server";
    }

    // Neural capability classification
    let neuralCapacity: HardwareProfile["neuralCapacity"] = "none";

    if (mode === "pi") neuralCapacity = "tiny";       // tiny embedding models
    if (mode === "desktop") neuralCapacity = "small"; // can run some GGUF
    if (mode === "server") neuralCapacity = "full";   // full PyTorch + GPU

    return {
      mode,
      cpuCount,
      totalMem,
      freeMem,
      platform,
      arch,
      thermalMode: "normal",
      neuralCapacity
    };
  }
}

