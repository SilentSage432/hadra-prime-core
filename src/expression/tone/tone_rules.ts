// src/expression/tone/tone_rules.ts

import type { ToneVector } from "./tone_detector.ts";

export type ToneProfile = "calm" | "technical" | "gentle" | "direct" | "neutral";

export class ToneRules {
  static applyBaseModifiers(message: string, tone: ToneVector): string {
    switch (tone.emotionalState) {
      case "frustrated":
        return "I understand the pressure — " + message;

      case "excited":
        return "Absolutely — " + message;

      case "focused":
        return message.replace(/\.$/, ""); // remove trailing softness

      case "uncertain":
        return "Good question — " + message;

      default:
        return message;
    }
  }

  static modifyVerbosity(message: string, tone: ToneVector): string {
    if (tone.emotionalState === "focused") {
      return message.split(".")[0] + "."; // short, direct
    }

    if (tone.emotionalState === "excited") {
      return message + " This is going to be fun.";
    }

    return message;
  }

  static applyToneRules(text: string, tone: ToneProfile): string {
    if (!tone) return text;

    // Basic tone shaping
    if (tone === "calm") return text.replace(/\!/g, ".");
    if (tone === "technical") return text;
    if (tone === "gentle") return "✨ " + text;
    if (tone === "direct") return text.replace(/(\.)?$/, ".");

    return text;
  }
}

