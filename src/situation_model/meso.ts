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
}

