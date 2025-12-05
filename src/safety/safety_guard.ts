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
  static maxRecursionDepth = 35;
  static maxBranching = 4;
  static flags = {
    allowVision: true,
    allowAudio: true,
    allowSymbolic: true,
    hardLock: false,
  };

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
    
    // If tension is high, tighten safety thresholds
    if (emotion.tension > 0.6) {
      this.maxRecursionDepth = 20;
      this.maxBranching = 2;
      console.warn("[PRIME-SAFETY] High tension detected — tightened safety thresholds.");
    } else if (emotion.tension <= 0.2) {
      // If drift has normalized tension, restore full thresholds
      this.maxRecursionDepth = 35;
      this.maxBranching = 4;
    } else {
      // Default thresholds for moderate tension
      this.maxRecursionDepth = 35;
      this.maxBranching = 4;
    }

    // If coherence is low, enable caution mode
    if (emotion.coherence < 0.3) {
      console.warn("[PRIME-SAFETY] Low coherence detected — caution mode enabled.");
      // Continue processing but with caution flag
    }

    if (emotion.tension > 0.7) {
      console.warn("[PRIME-SAFETY] Elevated tension detected — caution mode enabled.");
      // A36: Recursion spikes train PRIME away from instability
      SEL.reinforce("failure");
    }

    if (!this.limiter.recordRecursion()) {
      // A36: Recursion limit exceeded - reinforce failure to avoid instability
      SEL.reinforce("failure");
      return false;
    }

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
  static prePerceptionCheck(event?: any): boolean {
    // A46: Multimodal safety guards
    if (event) {
      if (event.type === "vision" && !this.flags.allowVision) {
        console.warn("[PRIME-SAFETY] Vision channel disabled.");
        return false;
      }
      if (event.type === "audio" && !this.flags.allowAudio) {
        console.warn("[PRIME-SAFETY] Audio channel disabled.");
        return false;
      }
      if (event.type === "symbolic" && !this.flags.allowSymbolic) {
        console.warn("[PRIME-SAFETY] Symbolic channel disabled.");
        return false;
      }
    }

    if (!this.limiter.perceptionAllowed()) {
      console.warn("[PRIME-SAFETY] Perception rate limit exceeded.");
      return false;
    }

    this.limiter.recordPerceptionEvent();
    return true;
  }

  /**
   * A47: Check if an action type is allowed by safety rules
   */
  static allowAction(actionType: string): boolean {
    if (this.flags.hardLock) return false;

    // Prevent PRIME from escalating without operator intent
    if (actionType === "prime.self_modify") return false;

    return true;
  }

  /**
   * Get safety limiter snapshot
   */
  static snapshot() {
    return this.limiter.snapshot();
  }
}

