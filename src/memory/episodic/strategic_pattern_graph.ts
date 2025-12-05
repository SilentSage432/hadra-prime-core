// src/memory/episodic/strategic_pattern_graph.ts
// A70: Strategic Pattern Graph Engine

import type { PatternCluster } from "./pattern_generalization_engine.ts";

export interface StrategicEdge {
  from: string;
  to: string;
  strength: number; // 0–1 likelihood weighting
  count: number; // how many times this transition has occurred
}

export class StrategicPatternGraph {
  private edges: StrategicEdge[] = [];

  // Record a transition: cluster A → cluster B
  recordTransition(from: string, to: string) {
    if (!from || !to || from === to) return;

    let edge = this.edges.find(e => e.from === from && e.to === to);

    if (!edge) {
      edge = { from, to, strength: 0, count: 0 };
      this.edges.push(edge);
    }

    edge.count += 1;
    // strength = normalized transition likelihood
    edge.strength = Math.min(1, edge.count / 20);

    return edge;
  }

  getEdges() {
    return [...this.edges];
  }

  // What does this cluster tend to lead to?
  getForwardLinks(clusterId: string) {
    return this.edges
      .filter(e => e.from === clusterId)
      .sort((a, b) => b.strength - a.strength);
  }

  // What tends to lead into this cluster?
  getBackLinks(clusterId: string) {
    return this.edges
      .filter(e => e.to === clusterId)
      .sort((a, b) => b.strength - a.strength);
  }

  // For UI: a simplified graph snapshot
  getGraphSnapshot() {
    return {
      nodes: Array.from(
        new Set(this.edges.flatMap(e => [e.from, e.to]))
      ),
      edges: this.edges
    };
  }
}

