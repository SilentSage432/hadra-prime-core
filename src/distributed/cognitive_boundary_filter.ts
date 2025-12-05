// src/distributed/cognitive_boundary_filter.ts
// A105: Identity Boundary & Cognitive Isolation Filter
// Prevents identity bleed and protects internal cognition

import { NodeIdentity } from "./node_identity.ts";
import { SAGE_IDENTITY } from "./sage_identity.ts";

// PRIME identity helper
const PRIME_IDENTITY = {
  get id() {
    return NodeIdentity.getIdentity().nodeId;
  },
  get role() {
    return NodeIdentity.getIdentity().role;
  }
};

export class CognitiveBoundaryFilter {
  static validateInbound(packet: any, origin: "SAGE" | "PRIME") {
    if (!packet || typeof packet !== "object") return null;

    // Identity verification
    if (origin === "SAGE" && packet.sender !== SAGE_IDENTITY.id) return null;
    if (origin === "PRIME" && packet.sender !== PRIME_IDENTITY.id) return null;

    // Drop unauthorized fields
    delete packet.internalState;
    delete packet.fullCognitionDump;

    return packet;
  }

  static validateOutbound(packet: any, destination: "SAGE" | "PRIME") {
    if (!packet || typeof packet !== "object") return null;

    // Prevent sending internal PRIME cognition upstream
    if (destination === "SAGE") {
      delete packet.metaSelf;
      delete packet.internalVoices;
    }

    return packet;
  }
}

