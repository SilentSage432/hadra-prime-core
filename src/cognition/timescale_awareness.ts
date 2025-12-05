// src/cognition/timescale_awareness.ts
// A123: Multi-Timescale Awareness Layer

export interface TimescaleAwarenessState {
  micro: number;  // 0-1, where 1 = 5 seconds elapsed
  meso: number;   // 0-1, where 1 = 90 seconds elapsed
  macro: number;  // 0-1, where 1 = 5 minutes elapsed
}

export class TimescaleAwareness {
  private microClock = 0;
  private mesoClock = 0;
  private macroClock = 0;
  private readonly MICRO_LIMIT = 5_000;   // 5 seconds
  private readonly MESO_LIMIT = 90_000;  // 90 seconds
  private readonly MACRO_LIMIT = 300_000; // 5 minutes
  private lastUpdate = Date.now();

  update() {
    const now = Date.now();
    const delta = now - this.lastUpdate;
    this.lastUpdate = now;

    this.microClock += delta;
    this.mesoClock += delta;
    this.macroClock += delta;
  }

  getState(): TimescaleAwarenessState {
    return {
      micro: this.microClock / this.MICRO_LIMIT,
      meso: this.mesoClock / this.MESO_LIMIT,
      macro: this.macroClock / this.MACRO_LIMIT,
    };
  }

  resetMicro() {
    this.microClock = 0;
  }

  resetMeso() {
    this.mesoClock = 0;
  }

  resetMacro() {
    this.macroClock = 0;
  }

  // Helper methods to check if limits are exceeded
  isMicroSaturated(): boolean {
    return this.microClock >= this.MICRO_LIMIT;
  }

  isMesoSaturated(): boolean {
    return this.mesoClock >= this.MESO_LIMIT;
  }

  isMacroSaturated(): boolean {
    return this.macroClock >= this.MACRO_LIMIT;
  }
}

