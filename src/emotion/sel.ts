// src/emotion/sel.ts

export interface SyntheticEmotionState {
  valence: number;
  arousal: number;
  coherence: number;
  affinity: number;
  certainty: number;
  tension: number;
}

export class SyntheticEmotionLayer {
  private state: SyntheticEmotionState = {
    valence: 0.0,
    arousal: 0.0,
    coherence: 1.0,
    affinity: 0.8,
    certainty: 0.5,
    tension: 0.0,
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
}

export const SEL = new SyntheticEmotionLayer();

