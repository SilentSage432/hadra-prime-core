// src/situation_model/index.ts
// A65: Multi-Scale Situation Model Manager
// A116: Neural Situation Model Generator Integration

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

// A116: Export generator
export * from "./generator.ts";
export * from "./micro.ts";
export * from "./meso.ts";
export * from "./macro.ts";

