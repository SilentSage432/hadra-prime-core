// src/planning/levels/atomic.ts

export class AtomicPlanner {
  requiresPermission = true;
  requiresYubiKey = false;

  decompose(tacticalSteps: any[]) {
    return tacticalSteps.map((step) => ({
      action: step,
      type: "internal",
      authorized: false
    }));
  }
}

