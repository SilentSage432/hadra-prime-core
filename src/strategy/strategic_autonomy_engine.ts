// A109 — Strategic Autonomy Engine
// PRIME's Level-1 Autonomous Reasoning Layer
// Enables strategic thought chains, multi-step reasoning, and outcome evaluation
// PRIME may reason autonomously, but may not act autonomously

import type { StrategicScenario, StrategicOutcome } from "./types.ts";
import { Knowledge } from "../cognition/knowledge/knowledge_graph.ts";
import { Recall } from "../cognition/recall_engine.ts";
import type { MetaReflectionEngine } from "../meta/meta_reflection_engine.ts";
import { generateEmbedding } from "../shared/embedding.ts";

export class StrategicAutonomyEngine {
  private knowledge: typeof Knowledge;
  private recall: typeof Recall;
  private meta: MetaReflectionEngine;

  constructor(
    knowledge: typeof Knowledge,
    recall: typeof Recall,
    meta: MetaReflectionEngine
  ) {
    this.knowledge = knowledge;
    this.recall = recall;
    this.meta = meta;
  }

  // Generate multi-step strategic plans
  generateStrategicChain(goal: string): StrategicScenario[] {
    // Get related concepts from knowledge graph
    const goalNode = this.knowledge.getNode(goal);
    const scenarios: StrategicScenario[] = [];

    if (goalNode) {
      // Get outgoing edges to find related concepts
      const edges = this.knowledge.getOutgoingEdges(goal);
      const relatedConcepts = edges
        .slice(0, 5)
        .map(edge => {
          const node = this.knowledge.getNode(edge.to);
          return node ? { id: node.id, label: node.id, weight: edge.weight } : null;
        })
        .filter((c): c is { id: string; label: string; weight: number } => c !== null);

      // Get memories related to the goal
      const goalEmbedding = generateEmbedding(16);
      const recallResults = this.recall.recall(goalEmbedding, 5);
      const memories = recallResults.map(r => 
        r.metadata?.summary || r.metadata?.type || "memory"
      );

      // Generate scenarios from related concepts
      for (const concept of relatedConcepts) {
        scenarios.push({
          goal,
          subgoal: concept.label,
          evidence: memories.slice(0, 3) // Use top 3 memories as evidence
        });
      }
    } else {
      // Fallback: create a basic scenario if goal not in knowledge graph
      const goalEmbedding = generateEmbedding(16);
      const recallResults = this.recall.recall(goalEmbedding, 3);
      const memories = recallResults.map(r => 
        r.metadata?.summary || r.metadata?.type || "memory"
      );

      scenarios.push({
        goal,
        subgoal: "explore",
        evidence: memories
      });
    }

    return scenarios;
  }

  // Score strategies based on alignment, clarity, and coherence
  scoreOutcome(scenario: StrategicScenario): StrategicOutcome {
    // Check goal alignment (simplified - uses meta reflection's clarity index)
    const alignment = this.computeAlignment(scenario.goal);
    
    // Clarity based on evidence quality
    const clarity = Math.min(scenario.evidence.length / 10, 1.0);
    
    // Coherence based on knowledge graph connections
    const coherence = this.computeCoherence(scenario.subgoal);
    
    // Combined score
    const score = (alignment * 0.4 + clarity * 0.3 + coherence * 0.3);
    
    return {
      ...scenario,
      score: Math.max(0, Math.min(1, score)) // Clamp between 0 and 1
    };
  }

  // Compute alignment score for a goal
  private computeAlignment(goal: string): number {
    // Check if goal exists in knowledge graph (indicates alignment)
    const node = this.knowledge.getNode(goal);
    if (node) {
      // Higher weight = better alignment
      return Math.min(node.weight, 1.0);
    }
    // Default moderate alignment for unknown goals
    return 0.5;
  }

  // Compute coherence score for a subgoal
  private computeCoherence(subgoal: string): number {
    const node = this.knowledge.getNode(subgoal);
    if (node) {
      // Check how well connected this node is
      const edges = this.knowledge.getEdges(subgoal);
      const connectionStrength = edges.length > 0
        ? edges.reduce((sum, e) => sum + e.weight, 0) / edges.length
        : 0.3;
      return Math.min(connectionStrength, 1.0);
    }
    // Default moderate coherence for unknown subgoals
    return 0.5;
  }

  // Main reasoning entry point
  reasonAbout(goal: string): StrategicOutcome[] {
    console.log(`[PRIME-STRATEGY] Evaluating strategic pathways for goal: ${goal}`);
    
    const chains = this.generateStrategicChain(goal);
    const scored = chains.map(c => this.scoreOutcome(c));
    const sorted = scored.sort((a, b) => b.score - a.score);
    
    if (sorted.length > 0) {
      console.log(`[PRIME-STRATEGY] Best outcome:`, {
        subgoal: sorted[0].subgoal,
        score: sorted[0].score.toFixed(3)
      });
    }
    
    return sorted;
  }
}

