// src/kernel/phase_scheduler.ts

export type CognitivePhase =
  | "INTERPRET"
  | "PREDICT"
  | "FUSE"
  | "REFLECT"
  | "SAFETY"
  | "MEMORY"
  | "HARMONIZE"
  | "COOL";

interface PhaseConfig {
  interval: number;       // ms between runs
  enabled: boolean;
  lastRun: number;
}

export class CognitivePhaseScheduler {
  private phases: Record<CognitivePhase, PhaseConfig> = {
    INTERPRET:  { interval: 300, enabled: true, lastRun: 0 },
    PREDICT:    { interval: 400, enabled: true, lastRun: 0 },
    FUSE:       { interval: 450, enabled: true, lastRun: 0 },
    REFLECT:    { interval: 700, enabled: true, lastRun: 0 },
    SAFETY:     { interval: 200, enabled: true, lastRun: 0 },
    MEMORY:     { interval: 500, enabled: true, lastRun: 0 },
    HARMONIZE:  { interval: 600, enabled: true, lastRun: 0 },
    COOL:       { interval: 1000, enabled: true, lastRun: 0 },
  };

  private activePhase: CognitivePhase | null = null;

  updatePhaseSpeed(phase: CognitivePhase, newInterval: number) {
    this.phases[phase].interval = newInterval;
  }

  disablePhase(phase: CognitivePhase) {
    this.phases[phase].enabled = false;
  }

  enablePhase(phase: CognitivePhase) {
    this.phases[phase].enabled = true;
  }

  tick(now: number): CognitivePhase[] {
    const ready: CognitivePhase[] = [];

    for (const key of Object.keys(this.phases) as CognitivePhase[]) {
      const cfg = this.phases[key];
      if (!cfg.enabled) continue;

      if (now - cfg.lastRun >= cfg.interval) {
        // Prevent re-entry of same or different phase during execution
        if (this.activePhase === null) {
          cfg.lastRun = now; // Update lastRun only when phase will actually run
          ready.push(key);
        }
      }
    }

    return ready;
  }

  beginPhase(phase: CognitivePhase) {
    this.activePhase = phase;
    // Update lastRun when phase actually begins execution
    if (this.phases[phase]) {
      this.phases[phase].lastRun = Date.now();
    }
  }

  endPhase() {
    this.activePhase = null;
  }
}

export const phaseScheduler = new CognitivePhaseScheduler();

