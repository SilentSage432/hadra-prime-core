/**
 * Safety Guard - Output filtering and content moderation
 */
import { SafetyLimiter } from "./safety_limiter/limiter.ts";
import { StabilityMatrix } from "../stability/stability_matrix.ts";
import { SEL } from "../emotion/sel.ts";

// Safety rate limiter - prevents runaway safety checks
let lastSafetyRun = 0;
const SAFETY_MIN_INTERVAL = 250; // ms

export class SafetyGuard {
  static limiter = new SafetyLimiter();

  /**
   * Filter and sanitize output text
   * Placeholder — future ML moderation layer goes here
   */
  filterOutput(text: string): string {
    // Placeholder — future ML moderation layer goes here
    return text.trim();
  }

  /**
   * Check if content is safe
   */
  isSafe(content: string): boolean {
    // Placeholder safety check
    return true;
  }

  /**
   * Pre-cognition safety check
   * Returns false if cognition should be blocked
   */
  static preCognitionCheck(): boolean {
    // Safety rate limiter - throttle safety checks
    const now = Date.now();
    if (now - lastSafetyRun < SAFETY_MIN_INTERVAL) {
      return true; // Allow by default when throttled
    }
    lastSafetyRun = now;

    // Emotion-aware safety check
    const emotion = SEL.getState();
    if (emotion.tension > 0.7) {
      console.warn("[PRIME-SAFETY] Elevated tension detected — caution mode enabled.");
      // Continue processing but with caution flag
    }

    if (!this.limiter.recordRecursion()) return false;

    if (!this.limiter.memoryAllowed()) {
      console.warn("[PRIME-SAFETY] Memory pressure too high.");
      return false;
    }

    if (StabilityMatrix.unstable()) {
      console.warn("[PRIME-SAFETY] Cognitive path denied due to instability.");
      return false;
    }

    return true;
  }

  /**
   * Pre-perception safety check
   * Returns false if perception event should be dropped
   */
  static prePerceptionCheck(): boolean {
    if (!this.limiter.perceptionAllowed()) {
      console.warn("[PRIME-SAFETY] Perception rate limit exceeded.");
      return false;
    }

    this.limiter.recordPerceptionEvent();
    return true;
  }

  /**
   * Get safety limiter snapshot
   */
  static snapshot() {
    return this.limiter.snapshot();
  }
}

