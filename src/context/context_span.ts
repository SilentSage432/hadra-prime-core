// src/context/context_span.ts

import { ContextNode } from "./context_node.ts";

export class ContextSpan {
  public nodes: ContextNode[] = [];
  public createdAt = Date.now();
  public updatedAt = Date.now();
  public label: string;

  constructor(label: string) {
    this.label = label;
  }

  addNode(node: ContextNode) {
    this.nodes.push(node);
    this.updatedAt = Date.now();
  }

  getActiveNodes(): ContextNode[] {
    return this.nodes.filter(n => !n.isExpired());
  }
}

