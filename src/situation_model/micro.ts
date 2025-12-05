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

  // A118: Check if micro situation is currently active
  isActive(): boolean {
    const age = Date.now() - this.state.timestamp;
    // Micro situations are active if they're less than 10 seconds old
    return age < 10000 && (this.state.activeGoal !== null || this.state.lastOperatorCommand !== null);
  }

  // A118: Generate human-readable summary
  summary(): string {
    if (this.state.activeGoal) {
      return `working on ${this.state.activeGoal}`;
    }
    if (this.state.lastOperatorCommand) {
      return `processing ${this.state.lastOperatorCommand}`;
    }
    return "idle";
  }

  // A119: Compute drift metric for micro situation
  drift(): number {
    // Default drift metric — can be overridden in future phases
    // For now, compute based on clarity and recency
    const age = Date.now() - this.state.timestamp;
    const ageDrift = Math.min(1, age / 30000); // 30 seconds = max drift
    const clarityDrift = 1 - this.state.clarity;
    return Math.max(ageDrift * 0.5, clarityDrift * 0.3, Math.random() * 0.2);
  }
}

