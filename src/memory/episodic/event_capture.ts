// src/memory/episodic/event_capture.ts
// A67: Episodic Memory - Event Capture
// A115: Event Boundary support for Neural Event Segmentation

export interface MicroEvent {
  timestamp: number;
  type: string;
  stability: any; // StabilitySnapshot from StabilityMatrix
  motivation: any; // MotivationState from MotivationEngine
  reflection: any | null; // CognitiveSummary from ReflectionEngine
  metadata?: Record<string, any>;
}

/**
 * A115: Event Boundary
 * Marks where one cognitive event ends and another begins
 */
export interface EventBoundary {
  id: string;
  timestamp: number;
  embedding: number[];
  intent: string;
  emotion: any;
  predictionError: number;
  windowRef: any; // Temporal window snapshot
  goal?: string | null;
  delta?: number;
  emotionShift?: number;
  reason?: string; // Why this boundary was detected
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

