// src/meta/meta_reflection_engine.ts

import { SelfModelVector } from "./self_model_vector.ts";
import { MetaLearningLayer } from "./meta_learning_layer.ts";
import { DriftPredictor } from "./drift_predictor.ts";
import { PredictiveMetaLearner } from "./predictive_meta_learner.ts";

export class MetaReflectionEngine {
  private smv: SelfModelVector;
  private metaLearner: MetaLearningLayer;
  private driftPredictor: DriftPredictor;
  private predictive: PredictiveMetaLearner;

  constructor(smv: SelfModelVector, metaLearner: MetaLearningLayer, driftPredictor: DriftPredictor) {
    this.smv = smv;
    this.metaLearner = metaLearner;
    this.driftPredictor = driftPredictor;
    this.predictive = new PredictiveMetaLearner();
  }

  runMetaCycle(goalSummary: any, motivation: any) {
    console.log("[PRIME-META] Running meta-reflection cycle...");

    // 1. Update PRIME's self-model vector based on current states
    this.smv.updateFromCycle(goalSummary, motivation);

    // 2. Predict cognitive drift
    const driftRisk = this.driftPredictor.evaluate(this.smv);

    if (driftRisk > 0.25) {
      console.log(`[PRIME-META] Drift detected (risk=${driftRisk.toFixed(3)}). Suggesting clarity pivot.`);
    }

    // 3. Apply meta-learning adjustments
    this.metaLearner.updateStrategies(this.smv, driftRisk);

    // ---- A55: PREDICTIVE META-LEARNING LAYER ----
    const forecast = this.predictive.forecast(this.smv);

    console.log(
      `[PRIME-PREDICT] Forecast → clarity=${forecast.projectedClarity.toFixed(3)}, ` +
      `drift=${forecast.projectedDrift.toFixed(3)}, stability=${forecast.projectedStability.toFixed(3)}`
    );

    const insights = this.predictive.generateInsights(this.smv, forecast);
    insights.forEach(i => console.log("[PRIME-INSIGHT]", i));

    console.log(
      `[PRIME-META] Meta-state: stability=${this.smv.stabilityIndex.toFixed(3)} ` +
      `clarity=${this.smv.clarityIndex.toFixed(3)} drift=${driftRisk.toFixed(3)}`
    );
  }
}

