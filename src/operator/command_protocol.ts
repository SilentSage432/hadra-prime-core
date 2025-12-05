// src/operator/command_protocol.ts

export type OperatorCommand = {
  op: string;            // e.g. "diagnose", "scan", "query", "dispatch"
  target?: string;       // subsystem or node
  parameters?: any;      // additional args
  priority?: "low" | "normal" | "high";
  token?: string | null; // YubiKey authority token (placeholder)
};

export class CommandProtocol {
  /** Validate structural correctness */
  static validate(cmd: OperatorCommand) {
    if (!cmd.op || typeof cmd.op !== "string") {
      return { valid: false, reason: "missing-op" };
    }

    const allowedOps = ["diagnose", "query", "scan", "dispatch", "analyze"];

    if (!allowedOps.includes(cmd.op)) {
      return { valid: false, reason: "op-not-allowed" };
    }

    return { valid: true };
  }

  /** Convert operator command → PRIME intent */
  static toIntent(cmd: OperatorCommand) {
    return {
      type: `action.${cmd.op}`,
      confidence: 1.0,
      payload: {
        target: cmd.target || null,
        parameters: cmd.parameters || {},
        priority: cmd.priority || "normal",
      },
    };
  }
}

