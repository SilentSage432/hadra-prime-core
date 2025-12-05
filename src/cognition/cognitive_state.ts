// src/cognition/cognitive_state.ts
// A113: Extended with neural signal support
// A120: Extended with attentional drift regulation fields
// A121: Extended with adaptive attentional weight shifting
// A123: Extended with multi-timescale awareness
// A124: Extended with multi-layered uncertainty engine
// A125: Extended with cognitive risk mitigation engine

import type { ToneVector } from "../expression/tone/tone_detector.ts";
import type { MetaState } from "../meta/meta_state.ts";
import type { NeuralSignal, CoherenceResult } from "./neural_symbolic_coherence.ts";
import type { TimescaleAwarenessState } from "./timescale_awareness.ts";
import type { UncertaintyVector } from "./uncertainty/uncertainty_engine.ts";
import type { MitigationStrategy } from "./risk/risk_mitigation_engine.ts";

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
  // A123: Multi-Timescale Awareness fields
  timescaleState?: TimescaleAwarenessState;  // Awareness of cognitive time evolution
  // A124: Multi-Layered Uncertainty Engine fields
  uncertaintyVector?: UncertaintyVector;      // Multi-dimensional uncertainty vector
  uncertaintyScore?: number;                  // Aggregated uncertainty score (0-1)
  noiseLevel?: number;                        // Cognitive noise level
  predictionVariance?: number;                // Variance in predictions
  conceptDrift?: number;                      // Concept instability/drift
  ambiguity?: number;                          // Situational ambiguity
  relationalUncertainty?: number;             // Uncertainty in operator/social intent
  // A125: Cognitive Risk Mitigation Engine fields
  riskScore?: number;                         // Aggregated risk score (0-1)
  mitigationStrategy?: MitigationStrategy;   // Current mitigation strategy
  narrativeInstability?: number;              // Narrative fragmentation risk
  emotionDrift?: number;                     // Emotional instability/drift
}

