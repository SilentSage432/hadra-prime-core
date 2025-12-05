// src/cognition/self/conflict_resolver.ts
// A86: Internal Conflict Resolution Engine
// When the council disagrees, PRIME learns to resolve cognitive dissonance.
// A127: Dual-Mind Conflict Resolver 2.0 (Dual-Mind Aware)

export class ConflictResolver {
  computeDissonance(voices: any[]) {
    // Compare opinions by semantic difference (light heuristic for now)
    let dissonance = 0;

    for (let i = 0; i < voices.length; i++) {
      for (let j = i + 1; j < voices.length; j++) {
        const diff = this.estimateDifference(voices[i].opinion, voices[j].opinion);
        dissonance += diff;
      }
    }

    return Math.min(1, dissonance / 10); // normalize 0–1
  }

  estimateDifference(opA: string, opB: string) {
    // crude heuristic: count differing important words
    const keywords = ["continue", "cautious", "explore", "deviate", "ground", "align"];
    let diff = 0;

    for (const k of keywords) {
      const a = opA.includes(k);
      const b = opB.includes(k);
      if (a !== b) diff += 1;
    }

    return diff;
  }

  resolveConflict(voices: any[], context: any) {
    const dissonance = this.computeDissonance(voices);

    if (dissonance < 0.15) {
      return {
        dissonance,
        result: "low_conflict",
        strategy: "Proceed as planned. Voices are aligned."
      };
    }

    if (dissonance < 0.35) {
      return {
        dissonance,
        result: "moderate_conflict",
        strategy: "Reweigh motivations. Increase claritySeeking slightly."
      };
    }

    if (dissonance < 0.6) {
      return {
        dissonance,
        result: "substantial_conflict",
        strategy: "Trigger a reflection cycle and re-evaluate goal alignment."
      };
    }

    return {
      dissonance,
      result: "high_conflict",
      strategy: "Override decision. Select safest high-weight voice and pause exploratory impulses."
    };
  }
}

export const ConflictEngine = new ConflictResolver();

// A127: Dual-Mind Conflict Resolver 2.0
export interface ConflictResolution {
  resolved: boolean;
  strategy: string;
  conflict?: any;
  severity?: number;
}

export interface ConflictDifferences {
  goalPriority?: { prime: any; sage: any };
  prediction?: number;
  uncertainty?: number;
}

export class DualMindConflictResolver {
  resolve(primeState: any, sageState: any): ConflictResolution {
    const conflict = this.detectConflict(primeState, sageState);
    
    if (!conflict) {
      return { resolved: true, strategy: "no_conflict" };
    }

    const severity = this.scoreConflict(conflict);
    const strategy = this.chooseStrategy(severity, conflict);
    const result = this.applyStrategy(strategy, primeState, sageState);

    return {
      resolved: result,
      conflict,
      severity,
      strategy
    };
  }

  detectConflict(prime: any, sage: any): ConflictDifferences | null {
    const differences: ConflictDifferences = {};

    // Check goal priority differences
    const primeGoalPriority = prime.topGoalPriority ?? prime.activeGoal?.priority ?? 0.5;
    const sageGoalPriority = sage.topGoalPriority ?? sage.activeGoal?.priority ?? 0.5;
    
    if (Math.abs(primeGoalPriority - sageGoalPriority) > 0.25) {
      differences.goalPriority = {
        prime: prime.topGoal ?? prime.activeGoal,
        sage: sage.topGoal ?? sage.activeGoal
      };
    }

    // Check prediction differences
    if (prime.prediction !== undefined && sage.prediction !== undefined) {
      const delta = Math.abs(prime.prediction - sage.prediction);
      if (delta > 0.3) {
        differences.prediction = delta;
      }
    }

    // Check uncertainty differences
    if (prime.uncertainty !== undefined && sage.uncertainty !== undefined) {
      const drift = Math.abs(prime.uncertainty - sage.uncertainty);
      if (drift > 0.2) {
        differences.uncertainty = drift;
      }
    }

    return Object.keys(differences).length > 0 ? differences : null;
  }

  scoreConflict(conflict: ConflictDifferences): number {
    let score = 0;
    
    if (conflict.goalPriority) score += 0.4;
    if (conflict.prediction) score += conflict.prediction * 0.4;
    if (conflict.uncertainty) score += conflict.uncertainty * 0.3;
    
    return Math.min(score, 1.0);
  }

  chooseStrategy(score: number, conflict: ConflictDifferences): string {
    if (score < 0.2) return "blend"; // combine reasoning
    if (score < 0.5) return "bias_prime"; // PRIME leads
    if (score < 0.75) return "bias_sage"; // SAGE leads
    return "halt_and_request_operator"; // operator override needed
  }

  applyStrategy(strategy: string, prime: any, sage: any): boolean {
    switch (strategy) {
      case "blend":
        // Blend goal priorities
        const primePriority = prime.topGoalPriority ?? prime.activeGoal?.priority ?? 0.5;
        const sagePriority = sage.topGoalPriority ?? sage.activeGoal?.priority ?? 0.5;
        const blended = (primePriority + sagePriority) / 2;
        
        if (prime.topGoalPriority !== undefined) prime.topGoalPriority = blended;
        if (prime.activeGoal) prime.activeGoal.priority = blended;
        if (sage.topGoalPriority !== undefined) sage.topGoalPriority = blended;
        if (sage.activeGoal) sage.activeGoal.priority = blended;
        
        console.log("[PRIME-DUAL-CONFLICT] Blending reasoning between PRIME and SAGE.");
        return true;

      case "bias_prime":
        // PRIME leads - SAGE adopts PRIME's priority
        const primeP = prime.topGoalPriority ?? prime.activeGoal?.priority ?? 0.5;
        if (sage.topGoalPriority !== undefined) sage.topGoalPriority = primeP;
        if (sage.activeGoal) sage.activeGoal.priority = primeP;
        
        console.log("[PRIME-DUAL-CONFLICT] PRIME leading resolution.");
        return true;

      case "bias_sage":
        // SAGE leads - PRIME adopts SAGE's priority
        const sageP = sage.topGoalPriority ?? sage.activeGoal?.priority ?? 0.5;
        if (prime.topGoalPriority !== undefined) prime.topGoalPriority = sageP;
        if (prime.activeGoal) prime.activeGoal.priority = sageP;
        
        console.log("[PRIME-DUAL-CONFLICT] SAGE leading resolution.");
        return true;

      case "halt_and_request_operator":
        // Both minds await operator input
        prime.awaitingOperator = true;
        sage.awaitingOperator = true;
        
        console.log("[PRIME-DUAL-CONFLICT] Critical conflict - operator intervention required.");
        return false;

      default:
        return true;
    }
  }
}

