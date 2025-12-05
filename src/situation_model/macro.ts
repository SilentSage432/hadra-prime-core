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

  // A118: Check if macro situation is currently active
  isActive(): boolean {
    // Macro situations are always active if they have objectives or narrative
    return this.state.longTermObjectives.length > 0 || 
           this.state.internalNarrative !== null ||
           this.state.operatorArc !== null;
  }

  // A118: Generate human-readable summary
  summary(): string {
    if (this.state.internalNarrative) {
      return this.state.internalNarrative;
    }
    if (this.state.operatorArc) {
      return this.state.operatorArc;
    }
    if (this.state.longTermObjectives.length > 0) {
      return this.state.longTermObjectives[0];
    }
    return "no macro context";
  }

  // A119: Compute drift metric for macro situation
  drift(): number {
    // Default drift metric — can be overridden in future phases
    // Macro drift is low if objectives/narrative are present, higher if missing
    const hasContext = this.state.longTermObjectives.length > 0 || 
                       this.state.internalNarrative !== null ||
                       this.state.operatorArc !== null;
    return hasContext ? Math.random() * 0.15 : Math.random() * 0.25;
  }
}

