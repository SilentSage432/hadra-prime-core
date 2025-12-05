// src/distributed/predictive_consensus.ts

import { ConsensusMath } from "./consensus_math.ts";
import type { DistributedSnapshot } from "./state_snapshot.ts";
import { PredictiveCoherence } from "../cognition/predictive_coherence.ts";
import { predictionToVector } from "./prediction_vector.ts";

export class PredictiveConsensus {
  private static neighborPredictions: number[][] = [];

  static registerSnapshot(snapshot: DistributedSnapshot) {
    if (snapshot.state.predictionVector) {
      this.neighborPredictions.push(snapshot.state.predictionVector);
    }

    // Limit buffer
    if (this.neighborPredictions.length > 10) {
      this.neighborPredictions.shift();
    }
  }

  static computeConsensus() {
    const selfPred = this.getCurrentPredictionVector();

    if (!selfPred || selfPred.length === 0) {
      return { agreement: 0, consensus: [], samples: 0 };
    }

    if (this.neighborPredictions.length === 0) {
      return { agreement: 1, consensus: selfPred, samples: 0 };
    }

    const similarities = this.neighborPredictions.map(p =>
      ConsensusMath.cosineSimilarity(selfPred, p)
    );

    const avgAgreement =
      similarities.reduce((a, b) => a + b, 0) / similarities.length;

    const consensusVec = ConsensusMath.consensusVector([
      selfPred,
      ...this.neighborPredictions,
    ]);

    return {
      agreement: avgAgreement,
      consensus: consensusVec,
      samples: this.neighborPredictions.length,
    };
  }

  static getCurrentPredictionVector(): number[] {
    const consensus = PredictiveCoherence.computeConsensus();
    return predictionToVector(consensus);
  }
}

