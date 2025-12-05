// src/situation_model/macro.ts
// A65: Macro-Scale Situation Model

export interface MacroSituation {
  longTermObjectives: string[];
  federationContext: string | null;
  operatorArc: string | null;
  cognitiveMilestones: string[];
  internalNarrative: string | null;
}

export class MacroSituationModel {
  private state: MacroSituation = {
    longTermObjectives: [],
    federationContext: null,
    operatorArc: null,
    cognitiveMilestones: [],
    internalNarrative: null
  };

  update(partial: Partial<MacroSituation>) {
    this.state = { ...this.state, ...partial };
  }

  snapshot(): MacroSituation {
    return { ...this.state };
  }
}

