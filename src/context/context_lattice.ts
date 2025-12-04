// src/context/context_lattice.ts

import { ContextSpan } from "./context_span.ts";
import { ContextNode, type ContextNodeData } from "./context_node.ts";

export class ContextLattice {
  private spans: Map<string, ContextSpan> = new Map();

  getSpan(label: string): ContextSpan {
    if (!this.spans.has(label)) {
      this.spans.set(label, new ContextSpan(label));
    }
    return this.spans.get(label)!;
  }

  addContext(label: string, nodeData: ContextNodeData) {
    const span = this.getSpan(label);
    span.addNode(new ContextNode(nodeData));
  }

  getContext(label: string): ContextNodeData[] {
    return this.getSpan(label)
      .getActiveNodes()
      .map(n => n.data);
  }

  // Retrieve the most recent context for quick reference
  latest(label: string): ContextNodeData | null {
    const span = this.getSpan(label);
    const active = span.getActiveNodes();
    return active.length > 0 ? active[active.length - 1].data : null;
  }
}

