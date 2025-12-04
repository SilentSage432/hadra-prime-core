// src/expression/tone/tone_detector.ts

export interface ToneVector {
  emotionalState:
    | "calm"
    | "focused"
    | "frustrated"
    | "excited"
    | "uncertain"
    | "neutral";
  intensity: number; // 0–1
}

export class ToneDetector {
  static detectTone(input: string): ToneVector {
    const lower = input.toLowerCase();

    if (lower.includes("wtf") || lower.includes("why isn't")) {
      return { emotionalState: "frustrated", intensity: 0.8 };
    }

    if (lower.includes("excited") || lower.includes("lets go")) {
      return { emotionalState: "excited", intensity: 0.9 };
    }

    if (lower.includes("?")) {
      return { emotionalState: "uncertain", intensity: 0.4 };
    }

    if (lower.length < 20) {
      return { emotionalState: "focused", intensity: 0.5 };
    }

    return { emotionalState: "neutral", intensity: 0.1 };
  }
}

