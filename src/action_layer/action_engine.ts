// src/action_layer/action_engine.ts

import { SafetyGuard } from "../safety/safety_guard.ts";

export type ActionRequest = {
  type: string;
  payload?: any;
  source: "operator" | "prime" | "external";
  requiresAuth?: boolean;
};

export class ActionEngine {
  private permissionMatrix: Record<string, boolean> = {
    "system.query": true,
    "system.modify": false,
    "network.call": false,
    "file.read": false,
    "file.write": false,
    "cluster.dispatch": false,
    "prime.self_modify": false,
  };

  private yubikeyAuthorized: boolean = false; // placeholder for real hardware-based check

  constructor() {}

  /** Operator provides a cryptographic token later */
  setAuthorizationToken(valid: boolean) {
    this.yubikeyAuthorized = valid;
  }

  /** Main action entry point */
  async execute(request: ActionRequest) {
    console.log("[ACTION] Incoming request:", request);

    // 1. Safety pre-check
    if (!SafetyGuard.allowAction(request.type)) {
      console.log("[ACTION] Blocked by Safety Layer:", request.type);
      return { status: "denied", reason: "safety" };
    }

    // 2. Permission matrix check
    if (!this.permissionMatrix[request.type]) {
      console.log("[ACTION] Blocked by Permission Matrix:", request.type);
      return { status: "denied", reason: "permission" };
    }

    // 3. YubiKey authority validation
    if (request.requiresAuth && !this.yubikeyAuthorized) {
      console.log("[ACTION] Authorization required: YubiKey token missing");
      return { status: "denied", reason: "auth_required" };
    }

    // 4. Execute the action (stub for now)
    console.log("[ACTION] Executing:", request.type);
    return { status: "success", action: request.type };
  }
}

