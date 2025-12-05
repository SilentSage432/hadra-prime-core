// src/strategy/strategy_engine.ts
// A64: Emergent Strategy Engine (Safe Edition)

import { MemoryStore } from "../memory/memory_store.ts";
import { SEL } from "../emotion/sel.ts";
import { SafetyGuard } from "../safety/safety_guard.ts";

export interface StrategyRequest {
  domain: string;
  goal: string;
  constraints?: string[];
  context?: Record<string, any>;
}

export interface StrategyPlan {
  steps: string[];
  risks: string[];
  rationale: string;
  operatorApprovalRequired: true;
  safe: true;
}

export class StrategyEngine {
  private memory: MemoryStore;
  private sel: typeof SEL;
  private safety: typeof SafetyGuard;

  constructor(memory: MemoryStore) {
    this.memory = memory;
    this.sel = SEL;
    this.safety = SafetyGuard;
  }

  /**
   * SAFE ENTRY POINT:
   * PRIME can ONLY generate strategies when the operator calls this directly.
   */
  generateStrategies(req: StrategyRequest): StrategyPlan[] {
    // Safety check: Block autonomous strategy generation
    if (!this.safety.enforceStrategyCall()) {
      throw new Error("Autonomous strategy generation is disallowed.");
    }

    const patterns = this.detectPatterns(req);
    const scenarios = this.generateScenarios(req, patterns);
    const ranked = this.rankScenarios(scenarios);
    return ranked.map((scenario) => ({
      steps: scenario.steps,
      risks: scenario.risks,
      rationale: scenario.rationale,
      operatorApprovalRequired: true,
      safe: true
    }));
  }

  /** Step 1: Detect relevant patterns in memory */
  private detectPatterns(req: StrategyRequest): any[] {
    // Use MemoryStore's retrieveRelevant method
    const relevant = this.memory.retrieveRelevant(req.domain);
    const goalRelevant = this.memory.retrieveRelevant(req.goal);
    
    // Combine and deduplicate
    const all = [...relevant, ...goalRelevant];
    const unique = Array.from(
      new Map(all.map((item: any, idx: number) => [item.id || idx, item])).values()
    );

    // Limit to 5 most relevant
    return unique.slice(0, 5).map((item: any) => ({
      summary: item.summary || item.intent?.type || "pattern",
      id: item.id,
      timestamp: item.timestamp
    }));
  }

  /** Step 2: Generate hypothetical pathways (non-autonomous) */
  private generateScenarios(req: StrategyRequest, patterns: any[]): any[] {
    if (patterns.length === 0) {
      // Fallback scenario if no patterns found
      return [{
        steps: [
          `Analyze domain: ${req.domain}`,
          `Identify key constraints: ${req.constraints?.join(", ") || "none"}`,
          `Propose pathway toward goal: ${req.goal}`
        ],
        risks: [
          "Requires operator approval",
          "No memory patterns found - using generic approach",
          "Scenario validity unvalidated"
        ],
        rationale: `No prior patterns found for domain '${req.domain}'. Proposing generic pathway toward goal '${req.goal}'.`
      }];
    }

    return patterns.map((p) => ({
      steps: [
        `Analyze domain: ${req.domain}`,
        `Identify key constraints: ${req.constraints?.join(", ") || "none"}`,
        `Leverage memory pattern: ${p.summary || "pattern"}`,
        `Propose pathway toward goal: ${req.goal}`
      ],
      risks: [
        "Requires operator approval",
        "Depends on accurate memory retrieval",
        "Scenario validity unvalidated"
      ],
      rationale: `Pattern '${p.summary}' suggests these steps may align with your goal '${req.goal}'.`
    }));
  }

  /** Step 3: Rank scenarios (simple heuristic, still safe) */
  private rankScenarios(scenarios: any[]): any[] {
    return scenarios.sort((a, b) =>
      a.risks.length - b.risks.length
    );
  }
}

