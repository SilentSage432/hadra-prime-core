// src/safety/safety_engine.ts

let lastSafetyCall = 0;
const SAFETY_COOLDOWN_MS = 200;

export function safetyEngine(state: any) {
  const now = Date.now();
  if (now - lastSafetyCall < SAFETY_COOLDOWN_MS) {
    return {
      ...state,
      safety: { halt: false }
    };
  }

  lastSafetyCall = now;

  if (state.recursionRisk > 5) {
    console.log("[PRIME-SAFETY] Recursion blocked.");
    return {
      ...state,
      safety: { halt: true }
    };
  }

  return {
    ...state,
    safety: { halt: false }
  };
}

