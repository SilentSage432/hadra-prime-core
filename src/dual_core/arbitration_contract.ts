// src/dual_core/arbitration_contract.ts
// A104b: Arbitration Contract
// Defines how PRIME + SAGE resolve conflicting intentions safely

export interface ArbitrationDecision {
  winner: "PRIME" | "SAGE" | "MERGE";
  reason: string;
}

export class ArbitrationContract {
  static decide(primePriority: number, sagePriority: number): ArbitrationDecision {
    if (primePriority > sagePriority) {
      return { winner: "PRIME", reason: "Higher computed priority" };
    }
    if (sagePriority > primePriority) {
      return { winner: "SAGE", reason: "Higher computed priority" };
    }
    return { winner: "MERGE", reason: "Equal priority — cooperative mode" };
  }
}

