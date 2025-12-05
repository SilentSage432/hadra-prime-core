// A116 — Joint Situation Modeler (JSM)
// Shared episodic awareness + shared situational understanding
// Merges PRIME's symbolic perspective with SAGE's telemetry + cluster state
// Creates a shared synthetic "now" that both minds can read, update, question, and reason about

import type { SituationSnapshot } from "./generator.ts";
import { TemporalWindow, type TemporalSnapshot } from "../temporal/window.ts";

export interface JointSituationSnapshot {
  timestamp: number;
  prime: any; // PRIME's cognitive context
  sage: any; // SAGE's system state
  coherence: number; // How aligned PRIME and SAGE's perspectives are
  narrative: string; // Unified situational narrative
  disagreements?: string[]; // Areas where PRIME and SAGE differ
  consensus?: any; // Unified consensus view
}

export class JointSituationModeler {
  private window: TemporalWindow;
  private current: JointSituationSnapshot | null = null;
  private snapshotHistory: JointSituationSnapshot[] = [];
  private maxHistorySize = 100;

  constructor(contextHorizonMs: number = 5000) {
    // Create temporal window with appropriate size
    // Convert ms to approximate entries (assuming ~1 entry per second)
    const maxEntries = Math.ceil(contextHorizonMs / 1000);
    this.window = new TemporalWindow(maxEntries);
  }

  /**
   * Update the joint situation model with PRIME and SAGE perspectives
   */
  update(primeContext: any, sageState: any): void {
    const coherence = this.computeCoherence(primeContext, sageState);
    const narrative = this.buildNarrative(primeContext, sageState);
    const disagreements = this.detectDisagreements(primeContext, sageState);
    const consensus = this.generateConsensus(primeContext, sageState, coherence);

    const merged: JointSituationSnapshot = {
      timestamp: Date.now(),
      prime: primeContext,
      sage: sageState,
      coherence,
      narrative,
      disagreements: disagreements.length > 0 ? disagreements : undefined,
      consensus
    };

    this.current = merged;

    // Store in snapshot history
    this.snapshotHistory.push(merged);
    if (this.snapshotHistory.length > this.maxHistorySize) {
      this.snapshotHistory.shift();
    }

    // Record in temporal window (using a simplified snapshot format)
    this.window.record({
      t: merged.timestamp,
      clarity: coherence, // Use coherence as clarity proxy
      consolidation: consensus ? 0.8 : 0.5,
      curiosity: 0.5, // Placeholder
      stability: coherence > 0.7 ? 0.9 : 0.6
    });

    console.log("[PRIME-JSM] Joint situation updated:", {
      coherence: coherence.toFixed(3),
      hasDisagreements: disagreements.length > 0,
      consensusGenerated: !!consensus
    });
  }

  /**
   * Get current joint situation snapshot
   */
  getCurrent(): JointSituationSnapshot | null {
    return this.current;
  }

  /**
   * Get history of joint situations (temporal window)
   */
  getHistory(): TemporalWindow {
    return this.window;
  }

  /**
   * Get recent joint situation snapshots
   */
  getRecentSnapshots(count: number = 10): JointSituationSnapshot[] {
    return this.snapshotHistory.slice(-count);
  }

  /**
   * Get all snapshots
   */
  getAllSnapshots(): JointSituationSnapshot[] {
    return [...this.snapshotHistory];
  }

  /**
   * Compute coherence between PRIME and SAGE perspectives
   */
  private computeCoherence(prime: any, sage: any): number {
    if (!prime || !sage) return 0.0;

    // Simple coherence computation based on:
    // 1. Goal alignment
    // 2. State similarity
    // 3. Temporal alignment

    let coherence = 0.5; // Base coherence

    // Check goal alignment
    const primeGoal = prime?.activeGoal?.type || prime?.recommendedFocus || null;
    const sageGoal = sage?.activeTask || sage?.currentObjective || null;
    if (primeGoal && sageGoal) {
      // Simple string similarity (can be enhanced with embeddings)
      if (primeGoal === sageGoal || primeGoal.includes(sageGoal) || sageGoal.includes(primeGoal)) {
        coherence += 0.2;
      }
    }

    // Check state similarity (stability, health, etc.)
    const primeStability = prime?.coherenceScore || prime?.stability || 0.5;
    const sageStability = sage?.stability || sage?.health || 0.5;
    const stabilityDiff = Math.abs(primeStability - sageStability);
    coherence += (1 - stabilityDiff) * 0.2;

    // Check temporal alignment (recent updates)
    const primeTime = prime?.timestamp || 0;
    const sageTime = sage?.timestamp || 0;
    if (primeTime > 0 && sageTime > 0) {
      const timeDiff = Math.abs(primeTime - sageTime);
      // If within 5 seconds, add coherence
      if (timeDiff < 5000) {
        coherence += 0.1;
      }
    }

    return Math.min(1.0, Math.max(0.0, coherence));
  }

  /**
   * Build unified narrative from PRIME and SAGE perspectives
   */
  private buildNarrative(prime: any, sage: any): string {
    const primeSummary = prime?.summary || 
                         prime?.narrative || 
                         prime?.recommendedFocus || 
                         "analyzing context";
    
    const sageSummary = sage?.summary || 
                       sage?.status || 
                       sage?.currentState || 
                       "operational";

    return `PRIME and SAGE jointly observe: ${primeSummary} / ${sageSummary}`;
  }

  /**
   * Detect disagreements between PRIME and SAGE perspectives
   */
  private detectDisagreements(prime: any, sage: any): string[] {
    const disagreements: string[] = [];

    if (!prime || !sage) return disagreements;

    // Check for goal misalignment
    const primeGoal = prime?.activeGoal?.type || prime?.recommendedFocus;
    const sageGoal = sage?.activeTask || sage?.currentObjective;
    if (primeGoal && sageGoal && primeGoal !== sageGoal) {
      disagreements.push(`Goal mismatch: PRIME focuses on "${primeGoal}" while SAGE targets "${sageGoal}"`);
    }

    // Check for stability assessment differences
    const primeStability = prime?.coherenceScore || prime?.stability;
    const sageStability = sage?.stability || sage?.health;
    if (primeStability !== undefined && sageStability !== undefined) {
      const diff = Math.abs(primeStability - sageStability);
      if (diff > 0.3) {
        disagreements.push(`Stability assessment differs: PRIME=${primeStability.toFixed(2)}, SAGE=${sageStability.toFixed(2)}`);
      }
    }

    // Check for threat/anomaly detection differences
    const primeThreats = prime?.salience?.filter((s: string) => s.includes("pressure") || s.includes("degradation")) || [];
    const sageThreats = sage?.anomalies || sage?.warnings || [];
    if (primeThreats.length > 0 && sageThreats.length === 0) {
      disagreements.push("PRIME detects threats that SAGE does not report");
    } else if (primeThreats.length === 0 && sageThreats.length > 0) {
      disagreements.push("SAGE reports anomalies that PRIME does not detect");
    }

    return disagreements;
  }

  /**
   * Generate consensus view from PRIME and SAGE perspectives
   */
  private generateConsensus(prime: any, sage: any, coherence: number): any {
    if (coherence < 0.5) {
      return null; // Too low coherence for consensus
    }

    return {
      timestamp: Date.now(),
      goals: this.mergeGoals(prime, sage),
      stability: this.mergeStability(prime, sage),
      threats: this.mergeThreats(prime, sage),
      focus: this.mergeFocus(prime, sage),
      coherence: coherence
    };
  }

  /**
   * Merge goals from PRIME and SAGE
   */
  private mergeGoals(prime: any, sage: any): string[] {
    const goals: string[] = [];
    
    const primeGoal = prime?.activeGoal?.type || prime?.recommendedFocus;
    const sageGoal = sage?.activeTask || sage?.currentObjective;
    
    if (primeGoal) goals.push(`PRIME: ${primeGoal}`);
    if (sageGoal && sageGoal !== primeGoal) goals.push(`SAGE: ${sageGoal}`);
    
    return goals;
  }

  /**
   * Merge stability assessments
   */
  private mergeStability(prime: any, sage: any): number {
    const primeStability = prime?.coherenceScore || prime?.stability || 0.5;
    const sageStability = sage?.stability || sage?.health || 0.5;
    return (primeStability + sageStability) / 2;
  }

  /**
   * Merge threat assessments
   */
  private mergeThreats(prime: any, sage: any): string[] {
    const threats: string[] = [];
    
    const primeThreats = prime?.salience?.filter((s: string) => 
      s.includes("pressure") || s.includes("degradation") || s.includes("threat")
    ) || [];
    const sageThreats = sage?.anomalies || sage?.warnings || [];
    
    threats.push(...primeThreats);
    threats.push(...sageThreats.map((t: any) => typeof t === "string" ? t : JSON.stringify(t)));
    
    return [...new Set(threats)]; // Remove duplicates
  }

  /**
   * Merge focus recommendations
   */
  private mergeFocus(prime: any, sage: any): string {
    const primeFocus = prime?.recommendedFocus || prime?.activeGoal?.type;
    const sageFocus = sage?.currentObjective || sage?.activeTask;
    
    if (primeFocus && sageFocus) {
      return primeFocus === sageFocus ? primeFocus : `${primeFocus} + ${sageFocus}`;
    }
    return primeFocus || sageFocus || "none";
  }
}

