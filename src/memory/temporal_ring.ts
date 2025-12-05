// src/memory/temporal_ring.ts

export interface TemporalEvent {
  ts: number;
  input?: any;
  fused?: any;
  intent?: any;
  output?: string;
  stabilityScore?: number;
}

export class TemporalRing {
  private static buffer: TemporalEvent[] = [];
  private static maxSize = 50; // last 50 cycles

  static push(event: TemporalEvent) {
    this.buffer.push(event);
    if (this.buffer.length > this.maxSize) {
      this.buffer.shift();
    }
  }

  static getRecent(n = 10) {
    return this.buffer.slice(-n);
  }

  static getAll() {
    return [...this.buffer];
  }

  static last() {
    return this.buffer[this.buffer.length - 1] ?? null;
  }
}

