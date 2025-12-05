// A111 — Internal Monologue Engine
// Structured, bounded, introspective self-dialogue loop
// PRIME's first inner voice - designed, interpretable, goal-driven

import type { CognitiveState } from "../cognitive_state.ts";
import { PRIME_LOOP_GUARD } from "../../shared/loop_guard.ts";

export interface DialogueTurn {
  thought: string;
  response: string;
  clarityDelta: number;
}

export interface MonologueResult {
  turns: DialogueTurn[];
  finalClarity: number;
  clarityImproved: boolean;
  halted: boolean;
  reason?: string;
}

export class InternalMonologueEngine {
  private maxTurns = 5;
  private clarityThreshold = 0.7;
  private clarityImprovementThreshold = 0.05; // Minimum improvement per turn

  /**
   * Run structured internal dialogue to refine decisions and improve clarity
   * Only runs when there's a goal and uncertainty signal
   */
  runDialogue(state: CognitiveState): MonologueResult | null {
    // Safety: Only run if there's a goal and need for reflection
    if (!state.activeGoal || !this.shouldRunDialogue(state)) {
      return null;
    }

    // Safety: Check loop guard to prevent runaway recursion
    if (!PRIME_LOOP_GUARD.enter()) {
      return {
        turns: [],
        finalClarity: state.uncertainty ? 1 - state.uncertainty : 0.5,
        clarityImproved: false,
        halted: true,
        reason: "Loop guard prevented recursion"
      };
    }

    try {
      let turns: DialogueTurn[] = [];
      let workingState = { ...state };
      let initialClarity = this.computeClarity(workingState);
      let clarityImproved = false;

      for (let turn = 0; turn < this.maxTurns; turn++) {
        // Form internal thought/question
        const thought = this.formThought(workingState, turn);
        
        // Generate internal response
        const response = this.formResponse(thought, workingState);
        
        // Compute clarity delta
        const beforeClarity = this.computeClarity(workingState);
        workingState = this.updateState(workingState, thought, response);
        const afterClarity = this.computeClarity(workingState);
        const clarityDelta = afterClarity - beforeClarity;

        turns.push({ thought, response, clarityDelta });

        // Check if clarity threshold reached
        if (afterClarity >= this.clarityThreshold) {
          clarityImproved = afterClarity > initialClarity;
          PRIME_LOOP_GUARD.exit();
          return {
            turns,
            finalClarity: afterClarity,
            clarityImproved,
            halted: false,
            reason: "Clarity threshold reached"
          };
        }

        // Check if clarity is improving
        if (clarityDelta < this.clarityImprovementThreshold && turn > 0) {
          // Clarity not improving enough, halt
          clarityImproved = afterClarity > initialClarity;
          PRIME_LOOP_GUARD.exit();
          return {
            turns,
            finalClarity: afterClarity,
            clarityImproved,
            halted: true,
            reason: "Clarity improvement insufficient"
          };
        }
      }

      // Max turns reached
      const finalClarity = this.computeClarity(workingState);
      clarityImproved = finalClarity > initialClarity;
      PRIME_LOOP_GUARD.exit();
      return {
        turns,
        finalClarity,
        clarityImproved,
        halted: true,
        reason: "Maximum turns reached"
      };
    } catch (error) {
      PRIME_LOOP_GUARD.exit();
      console.error("[PRIME-MONOLOGUE] Error during dialogue:", error);
      return null;
    }
  }

  /**
   * Determine if dialogue should run based on state
   */
  private shouldRunDialogue(state: CognitiveState): boolean {
    // Run if uncertainty is high or confidence is low
    const uncertainty = state.uncertainty ?? (1 - (state.confidence ?? 0.5));
    return uncertainty > 0.3; // Run if uncertainty > 30%
  }

  /**
   * Form an internal thought/question based on current state
   */
  private formThought(state: CognitiveState, turn: number): string {
    const goal = state.activeGoal?.type || "unknown";
    const uncertainty = state.uncertainty ?? (1 - (state.confidence ?? 0.5));

    if (turn === 0) {
      return `What is preventing clarity on goal '${goal}'?`;
    } else if (turn === 1) {
      return `What assumptions am I making about '${goal}'?`;
    } else if (turn === 2) {
      return `What evidence supports pursuing '${goal}'?`;
    } else if (turn === 3) {
      return `How does '${goal}' align with my current understanding?`;
    } else {
      return `What final clarification is needed for '${goal}'?`;
    }
  }

  /**
   * Generate internal response to thought
   */
  private formResponse(thought: string, state: CognitiveState): string {
    const uncertainty = state.uncertainty ?? (1 - (state.confidence ?? 0.5));
    const goal = state.activeGoal?.type || "unknown";

    if (uncertainty > 0.6) {
      return `Uncertainty is high (${uncertainty.toFixed(2)}); more context is needed for '${goal}'.`;
    } else if (uncertainty > 0.4) {
      return `Moderate uncertainty; reviewing assumptions about '${goal}' may help.`;
    } else if (uncertainty > 0.2) {
      return `Clarity improving; proceeding with primary strategy for '${goal}'.`;
    } else {
      return `Clarity sufficient; confident in approach to '${goal}'.`;
    }
  }

  /**
   * Update state based on dialogue turn
   */
  private updateState(
    state: CognitiveState,
    thought: string,
    response: string
  ): CognitiveState {
    // Reduce uncertainty based on dialogue (each turn improves clarity)
    const currentUncertainty = state.uncertainty ?? (1 - (state.confidence ?? 0.5));
    const reducedUncertainty = Math.max(0, currentUncertainty - 0.1);
    
    // Increase confidence correspondingly
    const newConfidence = Math.min(1, (state.confidence ?? 0.5) + 0.1);

    return {
      ...state,
      uncertainty: reducedUncertainty,
      confidence: newConfidence
    };
  }

  /**
   * Compute clarity score from state (inverse of uncertainty)
   */
  private computeClarity(state: CognitiveState): number {
    const uncertainty = state.uncertainty ?? (1 - (state.confidence ?? 0.5));
    return 1 - uncertainty;
  }
}

