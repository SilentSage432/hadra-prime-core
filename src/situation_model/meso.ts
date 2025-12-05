// src/situation_model/meso.ts
// A65: Meso-Scale Situation Model

export interface MesoSituation {
  trends: {
    clarityTrend: number;
    consolidationTrend: number;
    curiosityTrend: number;
  };
  memoryPressure: number;
  stabilityShift: number;
  cognitiveTrajectory: string; // passive label
}

export class MesoSituationModel {
  private state: MesoSituation = {
    trends: {
      clarityTrend: 0,
      consolidationTrend: 0,
      curiosityTrend: 0
    },
    memoryPressure: 0,
    stabilityShift: 0,
    cognitiveTrajectory: "stable",
  };

  update(partial: Partial<MesoSituation>) {
    this.state = { ...this.state, ...partial };
  }

  snapshot(): MesoSituation {
    return { ...this.state };
  }

  // A118: Check if meso situation is currently active
  isActive(): boolean {
    // Meso situations are active if there's meaningful cognitive trajectory
    return this.state.cognitiveTrajectory !== "stable" || 
           Math.abs(this.state.stabilityShift) > 0.1 ||
           this.state.memoryPressure > 0.2;
  }

  // A118: Generate human-readable summary
  summary(): string {
    const trajectory = this.state.cognitiveTrajectory;
    if (trajectory !== "stable") {
      return `${trajectory} cognitive state`;
    }
    if (this.state.memoryPressure > 0.5) {
      return "high memory pressure";
    }
    return "stable operation";
  }

  // A119: Compute drift metric for meso situation
  drift(): number {
    // Default drift metric — can be overridden in future phases
    // Compute based on stability shift and memory pressure
    const stabilityDrift = Math.abs(this.state.stabilityShift);
    const memoryDrift = this.state.memoryPressure;
    const trajectoryDrift = this.state.cognitiveTrajectory === "stable" ? 0 : 0.3;
    return Math.max(stabilityDrift, memoryDrift * 0.5, trajectoryDrift, Math.random() * 0.2);
  }
}

