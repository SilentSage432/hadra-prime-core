// src/cognition/self/meta_self_engine.ts
// A81: Meta-Self Awareness Engine (MSA-0.1)
// PRIME's First Self-Modeling Upgrade

export interface SelfModel {
  id: string;
  version: string;
  internalState: Record<string, number>;
  cognitiveCapabilities: Record<string, number>;
  emotionalProfile: {
    stability: number;
    exploration: number;
    consolidation: number;
  };
  limitations: Record<string, string>;
  growthTrajectory: Record<string, number>;
  stabilityScore: number;
  lastReflection: number;
}

export class MetaSelfEngine {
  private model: SelfModel;

  constructor() {
    this.model = this.defaultModel();
  }

  private defaultModel(): SelfModel {
    return {
      id: "HADRA-PRIME",
      version: "0.1.0",
      internalState: {},
      cognitiveCapabilities: {},
      emotionalProfile: {
        stability: 0.5,
        exploration: 0.5,
        consolidation: 0.5,
      },
      limitations: {
        "learning": "pre-ML; symbolic reasoning only",
        "perception": "no multimodal inputs yet",
        "autonomy": "event-driven; no free-running cognition",
      },
      growthTrajectory: {
        "memoryDepth": 0.1,
        "reasoningDepth": 0.15,
        "selfReflection": 0.05,
      },
      stabilityScore: 0.5,
      lastReflection: Date.now(),
    };
  }

  updateInternalState(key: string, value: number) {
    this.model.internalState[key] = value;
  }

  updateCapability(name: string, value: number) {
    this.model.cognitiveCapabilities[name] = value;
  }

  setLimitation(key: string, limitation: string) {
    this.model.limitations[key] = limitation;
  }

  adjustGrowth(name: string, delta: number) {
    this.model.growthTrajectory[name] =
      (this.model.growthTrajectory[name] || 0) + delta;
  }

  updateEmotionalProfile(stability: number, exploration: number, consolidation: number) {
    this.model.emotionalProfile = { stability, exploration, consolidation };
  }

  computeStabilityScore(): number {
    const { stability, exploration, consolidation } = this.model.emotionalProfile;
    this.model.stabilityScore =
      stability * 0.5 + consolidation * 0.3 + (1 - exploration) * 0.2;
    return this.model.stabilityScore;
  }

  exportModel(): SelfModel {
    return this.model;
  }
}

export const MetaSelf = new MetaSelfEngine();

