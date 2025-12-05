// src/distributed/node_identity.ts

import os from "os";
import crypto from "crypto";

export class NodeIdentity {
  private static nodeId: string = crypto.randomUUID();
  private static hostname: string = os.hostname();
  private static role: "primary" | "worker" | "edge" = "primary";

  static getIdentity() {
    return {
      nodeId: this.nodeId,
      hostname: this.hostname,
      role: this.role,
    };
  }

  static setRole(newRole: "primary" | "worker" | "edge") {
    this.role = newRole;
  }
}

