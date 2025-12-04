// src/expression/moderation/moderator.ts

import { StabilityMatrix } from "../../stability/stability_matrix.ts";
import { SafetyGuard } from "../../safety/safety_guard.ts";
import { ToneRules } from "../tone/tone_rules.ts";
import type { ToneVector } from "../tone/tone_detector.ts";

export type ToneProfile = "calm" | "technical" | "gentle" | "direct" | "neutral";

export class ExpressionModerator {
  static maxLength = 1200; // can adjust in future
  static minLength = 5;

  static enforceLength(text: string): string {
    if (text.length > this.maxLength) {
      return text.slice(0, this.maxLength) + "…";
    }
    return text;
  }

  static enforceSafety(text: string): string {
    const trimmed = text.trim();
    if (!trimmed) return "";

    // Prevent recursion or runaway text
    const safetySnapshot = SafetyGuard.snapshot();
    if (safetySnapshot.recursion > 10) {
      return "[PRIME] Output limited due to recursion safety.";
    }

    // Unstable system? Reduce detail.
    if (StabilityMatrix.unstable()) {
      return "[PRIME] System is stabilizing — reduced detail: " + trimmed.slice(0, 300);
    }

    return trimmed;
  }

  static applyTone(text: string, tone: ToneProfile): string {
    return ToneRules.applyToneRules(text, tone);
  }

  static moderate(text: string, tone: ToneProfile = "neutral"): string {
    let result = text;
    result = this.enforceLength(result);
    result = this.enforceSafety(result);
    result = this.applyTone(result, tone);
    return result;
  }
}

