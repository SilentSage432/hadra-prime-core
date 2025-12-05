// src/distributed/sage_identity.ts
// A104b: SAGE's distributed identity reference
// PRIME never impersonates this — only verifies it

const SAGE_NODE_ID = "SAGE-FEDERATION-PRIMARY";

export function getSageIdentity(): string {
  return SAGE_NODE_ID;
}

