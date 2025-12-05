// src/situation_model/multi_timescale_model.ts
// A118: Multi-Timescale Situation Modeling Layer
// A119: Contextual Attachment & Drift Detection Across Timescales

import { MicroSituationModel, type MicroSituation } from "./micro.ts";
import { MesoSituationModel, type MesoSituation } from "./meso.ts";
import { MacroSituationModel, type MacroSituation } from "./macro.ts";
import { DriftEngine, type DriftReport } from "./drift_engine.ts";

export interface TimescaleSituationState {
  micro: MicroSituationModel | null;
  meso: MesoSituationModel | null;
  macro: MacroSituationModel | null;
}

export class MultiTimescaleSituationModel {
  private state: TimescaleSituationState = {
    micro: null,
    meso: null,
    macro: null,
  };
  private driftEngine = new DriftEngine();
  private lastDriftReport: DriftReport | null = null;

  updateMicro(s: MicroSituationModel) {
    this.state.micro = s;
    return this.state;
  }

  updateMeso(s: MesoSituationModel) {
    this.state.meso = s;
    return this.state;
  }

  updateMacro(s: MacroSituationModel) {
    this.state.macro = s;
    return this.state;
  }

  getState(): TimescaleSituationState {
    return this.state;
  }

  // priorities: micro > meso > macro for immediate action
  resolveActiveSituation(): MicroSituationModel | MesoSituationModel | MacroSituationModel | null {
    if (this.state.micro?.isActive()) return this.state.micro;
    if (this.state.meso?.isActive()) return this.state.meso;
    return this.state.macro || null;
  }

  summarize() {
    return {
      micro: this.state.micro?.summary() || "none",
      meso: this.state.meso?.summary() || "none",
      macro: this.state.macro?.summary() || "none",
    };
  }

  // A119: Analyze drift across all timescales
  analyzeDrift(): DriftReport {
    const report = this.driftEngine.computeDrift(this.state);
    this.lastDriftReport = report;
    return report;
  }

  // A121: Get the last drift report for attentional weight computation
  getLastDrift(): DriftReport | null {
    return this.lastDriftReport;
  }
}

