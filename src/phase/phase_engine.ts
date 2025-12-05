// src/phase/phase_engine.ts

import { PRIME_LOOP_GUARD } from "../shared/loop_guard.ts";

export async function phaseEngine(state: any) {
  if (!PRIME_LOOP_GUARD.enter()) {
    console.log("[PRIME-PHASE] Blocked recursive entry");
    return state;
  }

  try {
    if (state.intent === null) {
      console.log("[PRIME-PHASE] Null intent — halting pipeline");
      return state;
    }

    if (state.safety?.halt) {
      console.log("[PRIME-PHASE] Safety halt engaged — stopping");
      return state;
    }

    return {
      ...state,
      phase: "complete",
    };
  } finally {
    PRIME_LOOP_GUARD.exit();
  }
}

