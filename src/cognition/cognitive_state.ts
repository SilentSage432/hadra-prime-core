// src/cognition/cognitive_state.ts

import type { ToneVector } from "../expression/tone/tone_detector.ts";
import type { MetaState } from "../meta/meta_state.ts";

export interface CognitiveState {
  intent: any;                // interpreted intent packet
  tone: ToneVector;           // tone vector from input
  context: any;               // context snapshot from ContextManager
  memory: any;                // recalled relevant memory
  priorityLevel: number;      // 0–1
  riskLevel: number;          // 0–1
  operatorFocus:              // inferred working mode
    | "build"
    | "debug"
    | "ideate"
    | "learn"
    | "unknown";
  recommendedResponseMode:    // structural suggestion
    | "direct"
    | "supportive"
    | "detailed"
    | "cautious";
  meta?: MetaState;           // meta-reasoning layer state
}

