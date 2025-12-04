// src/safety/safety_limiter/limiter.ts

export class SafetyLimiter {
  private recursionCount = 0;
  private recursionLimit = 20;
  private memoryPressure = 0; // 0–1
  private memoryLimit = 0.85;
  private perceptionRate = 0;
  private maxPerceptionRate = 50; // events/sec
  private perceptionWindowStart = Date.now();
  private perceptionWindowMs = 1000; // 1 second window

  recordRecursion(): boolean {
    this.recursionCount++;

    if (this.recursionCount > this.recursionLimit) {
      console.warn("[PRIME-SAFETY] Recursion limit exceeded. Halting branch.");
      return false;
    }

    return true;
  }

  resetRecursion() {
    this.recursionCount = 0;
  }

  setMemoryPressure(load: number) {
    this.memoryPressure = load;
  }

  memoryAllowed(): boolean {
    return this.memoryPressure < this.memoryLimit;
  }

  recordPerceptionEvent() {
    const now = Date.now();
    // Reset window if more than 1 second has passed
    if (now - this.perceptionWindowStart > this.perceptionWindowMs) {
      this.perceptionRate = 0;
      this.perceptionWindowStart = now;
    }
    this.perceptionRate++;
  }

  perceptionAllowed(): boolean {
    const now = Date.now();
    // Reset window if more than 1 second has passed
    if (now - this.perceptionWindowStart > this.perceptionWindowMs) {
      this.perceptionRate = 0;
      this.perceptionWindowStart = now;
    }
    return this.perceptionRate < this.maxPerceptionRate;
  }

  resetPerceptionWindow() {
    this.perceptionRate = 0;
    this.perceptionWindowStart = Date.now();
  }

  snapshot() {
    return {
      recursion: this.recursionCount,
      memoryPressure: this.memoryPressure,
      perceptionRate: this.perceptionRate,
    };
  }
}

