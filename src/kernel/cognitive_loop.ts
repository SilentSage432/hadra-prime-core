// src/kernel/cognitive_loop.ts

import { eventBus } from "./event_bus.ts";
import { SafetyGuard } from "../safety/safety_guard.ts";
import { predictionEngine } from "../interpretation/prediction_engine.ts";
import { PRIMEConfig } from "../shared/config.ts";

export class CognitiveLoop {
  private running = false;
  private cooldown = false;

  start() {
    if (this.running) return;
    this.running = true;

    // PRIME no longer loops — it listens.
    eventBus.on("cognitive-event", (evt) => this.process(evt));
    console.log("[PRIME-SCHEDULER] Cognitive loop initialized (event-driven).");
  }

  private async process(evt: any) {
    if (this.cooldown) return;

    // Safety — prevents runaway recursion chains
    if (!this.ensure()) {
      console.log("[PRIME-SCHEDULER] Entering cooldown mode…");
      this.cooldown = true;
      setTimeout(() => {
        this.cooldown = false;
        this.reset();
      }, 250); // 250ms cognitive reset window
      return;
    }

    // Process prediction if the event requests cognition
    if (evt.type === "think" && evt.payload?.allowPrediction) {
      const p = predictionEngine.generate({
        depth: "short",
        requireIntent: true,  // <- must have active intent to predict
      });
      console.log("[PRIME-PREDICT]", p);
    }

    // Reset stimulus flag after processing
    PRIMEConfig.runtime.hasStimulus = false;

    // Additional cognition modules can be added here
  }

  private ensure(): boolean {
    // Safety check wrapper
    return SafetyGuard.preCognitionCheck();
  }

  private reset() {
    // Reset safety limiter recursion counter
    SafetyGuard.limiter.resetRecursion();
  }
}

export const cognitiveLoop = new CognitiveLoop();

