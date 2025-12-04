export const IntentTypes = {
  OPERATOR: "operator",
  QUERY: "query",
  DIALOG: "dialog",
  ACTION: "action",
  UNKNOWN: "unknown",
} as const;

export type IntentType = typeof IntentTypes[keyof typeof IntentTypes];

export interface IntentPacket {
  type: IntentType;
  confidence: number;
  ruleScore: number;
  semanticScore: number;
  payload?: {
    command?: string;
    args?: string[];
  };
  raw: string;
  timestamp: Date;
}

