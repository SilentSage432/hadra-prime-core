// src/cognition/knowledge/knowledge_graph.ts
// A77: PRIME's Knowledge Graph Engine

export type NodeType =
  | "memory"
  | "concept"
  | "meta"
  | "domain"
  | "state"
  | "reflection"
  | "event";

export interface GraphNode {
  id: string;
  type: NodeType;
  weight: number;
  data?: any;
}

export interface GraphEdge {
  from: string;
  to: string;
  weight: number;
  relation: string;
}

export class KnowledgeGraph {
  private nodes = new Map<string, GraphNode>();
  private edges: GraphEdge[] = [];
  private maxEdges = 10000; // Prevent runaway growth

  addNode(id: string, type: NodeType, data?: any, weight = 1.0) {
    // Update if exists, otherwise create new
    this.nodes.set(id, { id, type, data, weight });
  }

  addEdge(
    from: string,
    to: string,
    weight: number,
    relation: string = "related"
  ) {
    // Prevent duplicate edges (same from, to, relation)
    const exists = this.edges.some(
      (e) => e.from === from && e.to === to && e.relation === relation
    );

    if (!exists) {
      this.edges.push({ from, to, weight, relation });
      this.enforceEdgeLimit();
    }
  }

  private enforceEdgeLimit() {
    if (this.edges.length > this.maxEdges) {
      // Keep highest weighted edges
      this.edges = this.edges
        .sort((a, b) => b.weight - a.weight)
        .slice(0, this.maxEdges);
      console.warn(`[PRIME-KNOWLEDGE] Edge limit reached, kept top ${this.maxEdges} edges`);
    }
  }

  getNode(id: string): GraphNode | undefined {
    return this.nodes.get(id);
  }

  getEdges(id: string): GraphEdge[] {
    return this.edges.filter((e) => e.from === id || e.to === id);
  }

  /** Get outgoing edges from a node */
  getOutgoingEdges(id: string): GraphEdge[] {
    return this.edges.filter((e) => e.from === id);
  }

  /** Get incoming edges to a node */
  getIncomingEdges(id: string): GraphEdge[] {
    return this.edges.filter((e) => e.to === id);
  }

  /** Link nodes based on similarity */
  linkSimilar(idA: string, idB: string, similarity: number) {
    if (similarity > 0.5) {
      this.addEdge(idA, idB, similarity, "similar");
    }
  }

  /** Link parent-child containment relationships */
  linkContainment(parent: string, child: string) {
    this.addEdge(parent, child, 1.0, "contains");
  }

  /** Link temporal/sequential relationships */
  linkSequence(before: string, after: string) {
    this.addEdge(before, after, 1.0, "precedes");
  }

  /** Link causal influence relationships */
  linkCausal(cause: string, effect: string, strength: number) {
    this.addEdge(cause, effect, strength, "causes");
  }

  /** Find nodes by type */
  getNodesByType(type: NodeType): GraphNode[] {
    return Array.from(this.nodes.values()).filter((n) => n.type === type);
  }

  /** Get graph statistics */
  getStats() {
    return {
      nodeCount: this.nodes.size,
      edgeCount: this.edges.length,
      nodesByType: Array.from(this.nodes.values()).reduce(
        (acc, n) => {
          acc[n.type] = (acc[n.type] || 0) + 1;
          return acc;
        },
        {} as Record<NodeType, number>
      ),
    };
  }

  /** Export full graph structure */
  exportGraph() {
    return {
      nodes: Array.from(this.nodes.values()),
      edges: [...this.edges],
    };
  }

  /** Clear the graph (use with caution) */
  clear() {
    this.nodes.clear();
    this.edges = [];
  }
}

export const Knowledge = new KnowledgeGraph();

