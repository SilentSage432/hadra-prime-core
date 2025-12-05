// src/dual_core/identity_exchange.ts
// A104b: Identity Exchange
// Handles identity validation between PRIME ↔ SAGE

import { getSageIdentity } from "../distributed/sage_identity.ts";
import { NodeIdentity } from "../distributed/node_identity.ts";
import { DualMindProtocol } from "./dual_mind_protocol.ts";

export function getPrimeNodeId(): string {
  return NodeIdentity.getIdentity().nodeId;
}

export class IdentityExchange {
  static perform() {
    const primeId = getPrimeNodeId();
    const sageId = getSageIdentity();
    return DualMindProtocol.initiateHandshake(primeId, sageId);
  }
}

