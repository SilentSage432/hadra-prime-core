// src/memory/episodic/event_capture.ts
// A67: Episodic Memory - Event Capture

export interface MicroEvent {
  timestamp: number;
  type: string;
  stability: any; // StabilitySnapshot from StabilityMatrix
  motivation: any; // MotivationState from MotivationEngine
  reflection: any | null; // CognitiveSummary from ReflectionEngine
  metadata?: Record<string, any>;
}

export class EventCapture {
  private buffer: MicroEvent[] = [];

  capture(event: Omit<MicroEvent, "timestamp">) {
    const record: MicroEvent = {
      timestamp: Date.now(),
      ...event
    };

    this.buffer.push(record);

    // Prevent runaway growth
    if (this.buffer.length > 5000) {
      this.buffer.shift();
    }

    return record;
  }

  flush() {
    const out = [...this.buffer];
    this.buffer = [];
    return out;
  }

  getBuffer() {
    return [...this.buffer];
  }
}

