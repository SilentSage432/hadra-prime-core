// src/kernel/event_bus.ts
// A104b: Dual-Mind Integration

import { EventEmitter } from "events";
import { DualMind } from "../dual_core/dual_mind_activation.ts";

class PrimeEventBus extends EventEmitter {
  emit(event: string | symbol, ...args: any[]): boolean {
    // A104b: Optional event mirroring or shared perception when dual-mind is active
    if (DualMind.isActive()) {
      // Optional: event mirroring or shared perception
      // Left intentionally lightweight so Cursor doesn't over-patch
    }
    return super.emit(event, ...args);
  }
}

export const eventBus = new PrimeEventBus();

// Utility wrappers
export function triggerCognition(allowPrediction = false) {
  eventBus.emit("cognitive-event", { 
    type: "think", 
    payload: { allowPrediction } 
  });
}

// Input event wrapper
export function triggerInput(input: string) {
  eventBus.emit("prime.input", { text: input });
}

