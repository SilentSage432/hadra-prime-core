// src/situation_model/micro.ts
// A65: Micro-Scale Situation Model

export interface MicroSituation {
  timestamp: number;
  lastOperatorCommand: string | null;
  emotionalState: Record<string, number>;
  safetyPressure: number;
  clarity: number;
  cognitiveLoad: number;
  activeGoal: string | null;
}

export class MicroSituationModel {
  private state: MicroSituation = {
    timestamp: Date.now(),
    lastOperatorCommand: null,
    emotionalState: {},
    safetyPressure: 0,
    clarity: 0,
    cognitiveLoad: 0,
    activeGoal: null,
  };

  update(partial: Partial<MicroSituation>) {
    this.state = { ...this.state, ...partial, timestamp: Date.now() };
  }

  snapshot(): MicroSituation {
    return { ...this.state };
  }
}

