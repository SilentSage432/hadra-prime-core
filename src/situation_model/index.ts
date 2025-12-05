// src/situation_model/index.ts
// A65: Multi-Scale Situation Model Manager

import { MicroSituationModel } from "./micro.ts";
import { MesoSituationModel } from "./meso.ts";
import { MacroSituationModel } from "./macro.ts";

export class SituationModel {
  micro = new MicroSituationModel();
  meso = new MesoSituationModel();
  macro = new MacroSituationModel();

  snapshot() {
    return {
      micro: this.micro.snapshot(),
      meso: this.meso.snapshot(),
      macro: this.macro.snapshot(),
    };
  }
}

export const PRIME_SITUATION = new SituationModel();

