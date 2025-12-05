// src/desire/proto_desire_engine.ts

export class ProtoDesireEngine {
  computeValenceChange(before: any, after: any) {
    let score = 0;

    // clarity improvement is desirable
    score += (after.clarityIndex - before.clarityIndex) * 0.6;

    // stability improvement is desirable
    score += (after.stabilityIndex - before.stabilityIndex) * 0.7;

    // drift reduction is desirable
    score += (before.driftRisk - after.driftRisk) * 0.8;

    // tension reduction is desirable
    score += (before.consolidationTension - after.consolidationTension) * 0.5;

    // Normalize to [-1, 1]
    if (score > 1) score = 1;
    if (score < -1) score = -1;

    return score;
  }

  updateProtoDesire(smv: any, valence: number) {
    if (!smv.desireState) {
      smv.desireState = {
        cumulativeValence: 0,
        recentValence: 0
      };
    }

    smv.desireState.recentValence = valence;
    smv.desireState.cumulativeValence += valence * 0.1; // slow accumulation

    // clamp
    if (smv.desireState.cumulativeValence > 1) smv.desireState.cumulativeValence = 1;
    if (smv.desireState.cumulativeValence < -1) smv.desireState.cumulativeValence = -1;

    return smv.desireState;
  }
}

