// src/shared/loop_guard.ts

export class LoopGuard {
  private active = false;

  enter() {
    if (this.active) return false;
    this.active = true;
    return true;
  }

  exit() {
    this.active = false;
  }
}

export const PRIME_LOOP_GUARD = new LoopGuard();

