// src/cognition/cognitive_state.ts
// A113: Extended with neural signal support
// A120: Extended with attentional drift regulation fields
// A121: Extended with adaptive attentional weight shifting

import type { ToneVector } from "../expression/tone/tone_detector.ts";
import type { MetaState } from "../meta/meta_state.ts";
import type { NeuralSignal, CoherenceResult } from "./neural_symbolic_coherence.ts";

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
  // A113: Neural signal integration
  neuralSignals?: NeuralSignal[];  // Array of neural signals
  neuralCoherence?: CoherenceResult | null;  // Interpreted coherence result
  temporalVector?: number[];  // A112: Temporal embedding vector
  // A120: Attention Drift Regulator fields
  focusBias?: number;              // Micro-level focus reinforcement
  distractionFilterStrength?: number;  // Resistance to distraction
  taskCommitment?: number;         // Meso-level task adherence
  contextAlignment?: number;       // Contextual coherence
  missionAlignment?: number;       // Macro-level mission alignment
  longHorizonFocus?: number;       // Long-term goal focus
  // A121: Adaptive Attentional Weight Shifting fields
  attentionMicro?: number;         // Micro-level attention weight (0-1)
  attentionMeso?: number;          // Meso-level attention weight (0-1)
  attentionMacro?: number;         // Macro-level attention weight (0-1)
}

