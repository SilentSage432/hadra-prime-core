// src/kernel/event_bus.ts

import { EventEmitter } from "events";

class PrimeEventBus extends EventEmitter {}

export const eventBus = new PrimeEventBus();

// Utility wrappers
export function triggerCognition(allowPrediction = false) {
  eventBus.emit("cognitive-event", { 
    type: "think", 
    payload: { allowPrediction } 
  });
}

