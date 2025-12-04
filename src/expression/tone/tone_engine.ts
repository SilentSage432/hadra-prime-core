// src/expression/tone/tone_engine.ts

import { ToneDetector } from "./tone_detector.ts";
import type { ToneVector } from "./tone_detector.ts";
import { ToneRules } from "./tone_rules.ts";
import { DefaultOperatorProfile } from "./tone_profiles.ts";
import type { OperatorProfile } from "./tone_profiles.ts";

export class ToneEngine {
  private profile: OperatorProfile = DefaultOperatorProfile;

  analyze(input: string): ToneVector {
    return ToneDetector.detectTone(input);
  }

  shapeOutput(message: string, tone: ToneVector): string {
    let shaped = ToneRules.applyBaseModifiers(message, tone);

    shaped = ToneRules.modifyVerbosity(shaped, tone);

    // Operator personalization
    if (!this.profile.prefersWarmth) {
      shaped = shaped.replace(/Absolutely — /, "");
      shaped = shaped.replace(/I understand the pressure — /, "");
    }

    return shaped;
  }
}

