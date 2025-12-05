// src/meta/stability_monitor.ts

export class StabilityMonitor {
  private instabilityCount = 0;
  private lastState = "";

  assess(state: string): boolean {
    if (state === this.lastState) {
      this.instabilityCount++;
    } else {
      this.instabilityCount = 0;
    }

    this.lastState = state;
    // If the same cognitive state fires too many times, it's unstable.
    return this.instabilityCount > 3;
  }

  reset() {
    this.instabilityCount = 0;
    this.lastState = "";
  }
}

