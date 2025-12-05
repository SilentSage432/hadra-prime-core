// src/dual_core/dual_mind_activation.ts
// A104b: Dual Mind Activation
// Main activation/deactivation controller for PRIME↔SAGE dual-mind system

import { IdentityExchange } from "./identity_exchange.ts";
import { SharedContextSeedBuilder } from "./shared_context_seed.ts";

export class DualMindActivation {
  private active = false;
  private seed: any = null;

  activate() {
    if (this.active) return;

    const handshake = IdentityExchange.perform();
    this.seed = SharedContextSeedBuilder.create(null, null);
    this.active = true;

    console.log("[DUAL-MIND] Activated.");
    console.log("[DUAL-MIND] Handshake:", handshake);
    console.log("[DUAL-MIND] Shared Context Seed:", this.seed);
  }

  deactivate() {
    if (!this.active) return;

    this.active = false;
    this.seed = null;
    console.log("[DUAL-MIND] Deactivated.");
  }

  isActive() {
    return this.active;
  }
}

export const DualMind = new DualMindActivation();

