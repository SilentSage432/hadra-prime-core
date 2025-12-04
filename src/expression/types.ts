export type ExpressionType =
  | "status"
  | "reply"
  | "warning"
  | "error"
  | "system";

export interface ExpressionPacket {
  type: ExpressionType;
  message: string;
  metadata?: Record<string, any>;
  confidence: number; // 0–1.0
}

