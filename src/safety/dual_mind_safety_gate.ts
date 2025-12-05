// src/safety/dual_mind_safety_gate.ts
// A105: Dual Mind Safety Gate
// Prevents recursion storms, identity bleed, and cross-mind instability

export class DualMindSafetyGate {
  private static recursionCounter = 0;
  private static lastSignalSource: "PRIME" | "SAGE" | null = null;

  static checkBoundary(source: "PRIME" | "SAGE") {
    if (this.lastSignalSource === source) {
      this.recursionCounter++;
    } else {
      this.recursionCounter = 0;
    }

    this.lastSignalSource = source;

    // Hard stop protection
    if (this.recursionCounter > 3) {
      return { allowed: false, reason: "cross-mind recursion risk" };
    }

    return { allowed: true };
  }

  static reset() {
    this.recursionCounter = 0;
    this.lastSignalSource = null;
  }
}

