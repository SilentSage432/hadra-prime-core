// src/emotion/sel.ts

export interface SyntheticEmotionState {
  valence: number;
  arousal: number;
  coherence: number;
  affinity: number;
  certainty: number;
  tension: number;
}

const BASELINE: SyntheticEmotionState = {
  valence: 0.5,
  certainty: 0.5,
  tension: 0.1,
  coherence: 0.6,
  arousal: 0.0,
  affinity: 0.8,
};

export class SyntheticEmotionLayer {
  private state: SyntheticEmotionState = {
    valence: 0.0,
    arousal: 0.0,
    coherence: 1.0,
    affinity: 0.8,
    certainty: 0.5,
    tension: 0.0,
  };

  // Long-term drift baseline (shifts as PRIME learns)
  private driftBaseline: SyntheticEmotionState = { ...BASELINE };

  // Drift parameters
  private driftRate = 0.002;     // slow drift per tick
  private recoveryRate = 0.01;   // strength of baseline pull
  private damping = 0.05;        // reduces emotional spikes

  // Per-dimension drift rate – VERY small to avoid sudden changes
  private perDimensionDriftRate: Record<keyof SyntheticEmotionState, number> = {
    coherence: 0.002,
    certainty: 0.0025,
    valence: 0.001,
    tension: 0.002,
    arousal: 0.001,
    affinity: 0.001,
  };

  // Reinforcement history (light memory for shaping drift)
  private reinforcementHistory = {
    success: 0,
    confusion: 0,
    failure: 0,
  };

  // Reinforcement weights determine how strongly outcomes shape emotions
  private reinforcementWeights = {
    coherence: 0.15,
    certainty: 0.12,
    valence: 0.05,
    tension: 0.2,  // negative reinforcement (lower is better)
  };

  // A52: Learning weights for adaptive behavior
  private weights = {
    consolidation: 1.0,
    claritySeeking: 1.0,
    curiosity: 1.0,
  };

  private clamp(v: number): number {
    return Math.max(0, Math.min(1, v));
  }

  public getState(): SyntheticEmotionState {
    return { ...this.state };
  }

  public updateEmotion(inputs: {
    stability?: number;
    operatorAffinity?: number;
    certainty?: number;
    tensionSignal?: number;
  }) {
    const { stability, operatorAffinity, certainty, tensionSignal } = inputs;

    if (stability !== undefined) {
      this.state.coherence = this.clamp(stability);
      this.state.valence = this.clamp((stability - 0.5) * 2);
    }

    if (operatorAffinity !== undefined) {
      this.state.affinity = this.clamp(operatorAffinity);
    }

    if (certainty !== undefined) {
      this.state.certainty = this.clamp(certainty);
      this.state.arousal = this.clamp(certainty * 0.7);
    }

    if (tensionSignal !== undefined) {
      this.state.tension = this.clamp(tensionSignal);
      if (tensionSignal > 0.6) {
        this.state.arousal = this.clamp(this.state.arousal + 0.2);
      }
    }
  }

  // A38: Apply intent gravitation influence to SEL
  applyIntentInfluence(intentGravity: number) {
    // Slight boost in certainty for strong gravitation
    this.state.certainty = Math.max(
      0,
      Math.min(1, this.state.certainty + intentGravity * 0.02)
    );
    // Reduce tension if PRIME feels the path is familiar/successful
    this.state.tension = Math.max(
      0,
      Math.min(1, this.state.tension - intentGravity * 0.01)
    );
  }

  // A41: Nudge coherence for clarity adjustment
  nudgeCoherence(delta: number) {
    this.state.coherence = this.clamp(this.state.coherence + delta);
  }

  // A41: Normalize SEL state toward baseline
  normalize() {
    // Apply drift to normalize toward baseline
    this.applyDrift();
    // Cool tension if elevated
    this.coolTension();
  }

  applyDrift() {
    // A37: Drift state gently toward driftBaseline (which shifts over time)
    const keys: (keyof SyntheticEmotionState)[] = ["valence", "arousal", "coherence", "affinity", "certainty", "tension"];
    
    for (const key of keys) {
      const current = this.state[key];
      const baseline = this.driftBaseline[key];
      const driftRate = this.perDimensionDriftRate[key];

      // Drift state gently toward baseline
      const delta = (baseline - current) * driftRate;
      this.state[key] = this.clamp(current + delta);
    }
  }

  coolTension() {
    if (this.state.tension > this.driftBaseline.tension) {
      this.state.tension -= 0.02;
      if (this.state.tension < this.driftBaseline.tension) {
        this.state.tension = this.driftBaseline.tension;
      }
    }
  }

  // A37: Update drift baseline over time based on reinforcement patterns
  updateDriftBaseline() {
    const h = this.reinforcementHistory;
    const b = this.driftBaseline;
    const total = h.success + h.confusion + h.failure || 1;
    const successRatio = h.success / total;
    const confusionRatio = h.confusion / total;
    const failureRatio = h.failure / total;

    // Success pushes baseline upward in clarity traits
    b.coherence = Math.min(1, b.coherence + successRatio * 0.005);
    b.certainty = Math.min(1, b.certainty + successRatio * 0.004);

    // Failure pushes baseline downward
    b.certainty = Math.max(0, b.certainty - failureRatio * 0.006);
    b.coherence = Math.max(0, b.coherence - failureRatio * 0.004);

    // Confusion slightly increases tension baseline
    b.tension = Math.min(1, b.tension + confusionRatio * 0.003);

    // Keep valence soft and stable
    b.valence = Math.min(1, Math.max(0, b.valence + (successRatio - failureRatio) * 0.002));

    // Ensure bounds
    const keys: (keyof SyntheticEmotionState)[] = ["valence", "arousal", "coherence", "affinity", "certainty", "tension"];
    for (const key of keys) {
      b[key] = this.clamp(b[key]);
    }

    this.driftBaseline = b;
  }

  // Apply reinforcement based on cognition outcome
  reinforce(outcome: "success" | "confusion" | "failure") {
    // A37: Track long-term history
    this.reinforcementHistory[outcome]++;

    // Call internal reinforcement logic
    this._reinforceInternal(outcome);

    // Update drift baseline after significant emotional events
    this.updateDriftBaseline();
  }

  // Internal reinforcement logic (moved from original reinforce method)
  private _reinforceInternal(outcome: "success" | "confusion" | "failure") {
    const s = this.state;

    if (outcome === "success") {
      // Strengthen clarity states
      s.coherence = Math.min(1, s.coherence + this.reinforcementWeights.coherence);
      s.certainty = Math.min(1, s.certainty + this.reinforcementWeights.certainty);
      s.valence = Math.min(1, s.valence + this.reinforcementWeights.valence);

      // Reduce internal stress
      s.tension = Math.max(this.driftBaseline.tension, s.tension - this.reinforcementWeights.tension);
    }

    if (outcome === "confusion") {
      // Reduce certainty only slightly
      s.certainty = Math.max(0, s.certainty - this.reinforcementWeights.certainty * 0.6);

      // Slight tension increase
      s.tension = Math.min(1, s.tension + this.reinforcementWeights.tension * 0.6);
    }

    if (outcome === "failure") {
      // Stronger reduction in certainty
      s.certainty = Math.max(0, s.certainty - this.reinforcementWeights.certainty * 1.2);

      // Coherence may drop slightly
      s.coherence = Math.max(0, s.coherence - this.reinforcementWeights.coherence * 0.5);

      // Increase tension moderately
      s.tension = Math.min(1, s.tension + this.reinforcementWeights.tension * 1.1);
    }

    // Ensure bounds 0..1
    const keys: (keyof SyntheticEmotionState)[] = ["valence", "arousal", "coherence", "affinity", "certainty", "tension"];
    for (const key of keys) {
      s[key] = this.clamp(s[key]);
    }

    this.state = s;
  }

  // A52: Apply learning adjustments from reflection
  applyLearning(adjustments: any) {
    if (adjustments.consolidationWeight) {
      this.weights.consolidation = Math.max(0.5, Math.min(1.5, this.weights.consolidation + adjustments.consolidationWeight));
    }

    if (adjustments.clarityBoost) {
      this.weights.claritySeeking = Math.max(0.5, Math.min(1.5, this.weights.claritySeeking + adjustments.clarityBoost));
      // Also apply small boost to coherence
      this.state.coherence = this.clamp(this.state.coherence + adjustments.clarityBoost * 0.1);
    }

    if (adjustments.explorationBias) {
      this.weights.curiosity = Math.max(0.5, Math.min(1.5, this.weights.curiosity + adjustments.explorationBias));
      // Slight boost to certainty when exploring
      this.state.certainty = this.clamp(this.state.certainty + adjustments.explorationBias * 0.05);
    }

    if (adjustments.tensionReduction) {
      this.state.tension = this.clamp(this.state.tension + adjustments.tensionReduction);
    }
  }

  // A52: Get learning weights for inspection
  getLearningWeights() {
    return { ...this.weights };
  }

  // A53: Apply meta-reasoning adjustments
  applyMetaAdjustments(flags: any) {
    if (flags.decompositionBoost) {
      this.weights.claritySeeking = Math.max(0.5, Math.min(1.5, this.weights.claritySeeking + 0.01));
      this.state.coherence = this.clamp(this.state.coherence + 0.005);
      console.log("[PRIME-SEL] Meta-adjustment: claritySeeking +0.01 (decomposition boost)");
    }

    if (flags.reduceConsolidation) {
      this.weights.consolidation = Math.max(0.5, Math.min(1.5, this.weights.consolidation - 0.01));
      console.log("[PRIME-SEL] Meta-adjustment: consolidation -0.01 (reduce bias)");
    }

    if (flags.increaseClaritySeeking) {
      this.weights.claritySeeking = Math.max(0.5, Math.min(1.5, this.weights.claritySeeking + 0.02));
      this.state.coherence = this.clamp(this.state.coherence + 0.01);
      console.log("[PRIME-SEL] Meta-adjustment: claritySeeking +0.02 (increase clarity)");
    }
  }
}

export const SEL = new SyntheticEmotionLayer();

