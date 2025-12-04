// src/stability/stability_monitor.ts

export interface SubsystemHealth {
  latency: number;
  errors: number;
  load: number; // 0–1 normalized
}

export class StabilityMonitor {
  private subsystems: Record<string, SubsystemHealth> = {};
  private history: any[] = [];

  registerSubsystem(name: string) {
    this.subsystems[name] = { latency: 0, errors: 0, load: 0 };
  }

  updateSubsystem(
    name: string,
    metrics: Partial<SubsystemHealth>
  ) {
    if (!this.subsystems[name]) return;

    this.subsystems[name] = {
      ...this.subsystems[name],
      ...metrics,
    };

    this.history.push({ ts: Date.now(), name, metrics });
  }

  computeStabilityScore(): number {
    const values = Object.values(this.subsystems);
    if (values.length === 0) return 1;

    const avgLoad =
      values.reduce((a, b) => a + b.load, 0) / values.length;
    const avgErrors =
      values.reduce((a, b) => a + b.errors, 0) / values.length;

    // Score: 1 = perfect, 0 = unstable
    return Math.max(
      0,
      1 - (avgLoad * 0.6 + avgErrors * 0.4)
    );
  }

  isUnstable(): boolean {
    return this.computeStabilityScore() < 0.4;
  }

  snapshot() {
    return {
      score: this.computeStabilityScore(),
      subsystems: this.subsystems,
    };
  }
}

