// src/context/context_node.ts

export interface ContextNodeData {
  type: string;             // e.g., "phase", "repo", "error", "topic", "emotion"
  value: any;               // flexible payload
  timestamp: number;        // ms epoch
  ttl?: number;             // optional time-to-live
}

export class ContextNode {
  public data: ContextNodeData;

  constructor(data: ContextNodeData) {
    this.data = data;
  }

  isExpired(): boolean {
    if (!this.data.ttl) return false;
    return Date.now() > this.data.timestamp + this.data.ttl;
  }
}

