// src/meta/meta_reflection_engine.ts

import { SelfModelVector } from "./self_model_vector.ts";
import { MetaLearningLayer } from "./meta_learning_layer.ts";
import { DriftPredictor } from "./drift_predictor.ts";
import { PredictiveMetaLearner } from "./predictive_meta_learner.ts";
import { SelfHealingEngine } from "../stability/self_healing_engine.ts";
import { AdaptiveReconfigEngine } from "../architecture/adaptive_reconfig_engine.ts";
import { SCELExpansionEngine } from "../expansion/scel_engine.ts";

export class MetaReflectionEngine {
  private smv: SelfModelVector;
  private metaLearner: MetaLearningLayer;
  private driftPredictor: DriftPredictor;
  private predictive: PredictiveMetaLearner;
  private healer: SelfHealingEngine;
  private aire: AdaptiveReconfigEngine;
  private scel: SCELExpansionEngine;
  private cognitiveModules: any[] = [];

  constructor(smv: SelfModelVector, metaLearner: MetaLearningLayer, driftPredictor: DriftPredictor) {
    this.smv = smv;
    this.metaLearner = metaLearner;
    this.driftPredictor = driftPredictor;
    this.predictive = new PredictiveMetaLearner();
    this.healer = new SelfHealingEngine();
    this.aire = new AdaptiveReconfigEngine();
    this.scel = new SCELExpansionEngine();
  }

  runMetaCycle(goalSummary: any, motivation: any) {
    console.log("[PRIME-META] Running meta-reflection cycle...");

    // 1. Update PRIME's self-model vector based on current states
    this.smv.updateFromCycle(goalSummary, motivation);

    // 2. Predict cognitive drift
    const driftRisk = this.driftPredictor.evaluate(this.smv);
    this.smv.driftRisk = driftRisk; // A56: Store drift risk for healing assessment

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

    // --- A56: RUN SELF-HEALING AFTER META-REFLECTION ---
    const health = this.healer.assessHealth(this.smv);
    
    if (health.clarityDrop || health.highDrift || health.unstable || health.overloaded) {
      console.log("[PRIME-HEAL] Healing pass triggered.");
      this.healer.microRepair(this.smv);
    }

    // Example regeneration trigger
    if (this.smv.clarityIndex < 0.38) {
      this.healer.regenerateSubsystem("clarity-lattice");
    }

    // --- A57: AIRE RUNTIME RECONFIG ---
    const reconfig = this.aire.analyze(this.smv);
    if (Object.values(reconfig).some(v => v === true)) {
      console.log("[AIRE] Reconfiguration triggered.");
      this.aire.applyReconfiguration(this.smv);
    }

    // Fallback regeneration
    if (this.smv.stabilityIndex < 0.42) {
      this.aire.regeneratePathway("stability-coherence");
    }

    // --- A58: SCEL: SELF-DIRECTED COGNITIVE EXPANSION ---
    const need = this.scel.detectNeed(this.smv);
    if (Object.values(need).some(v => v === true)) {
      console.log("[SCEL] Cognitive expansion opportunity detected.");

      let type =
        need.needsClarityModule ? "clarity" :
        need.needsDriftCorrector ? "drift" :
        need.needsFusionEnhancer ? "fusion" :
        "intent";

      const blueprint = this.scel.generateBlueprint(type);
      const before = {
        clarityIndex: this.smv.clarityIndex,
        driftRisk: this.smv.driftRisk,
        consolidationTension: this.smv.consolidationTension
      };

      blueprint.compute(this.smv);

      const after = {
        clarityIndex: this.smv.clarityIndex,
        driftRisk: this.smv.driftRisk,
        consolidationTension: this.smv.consolidationTension
      };

      const ok = this.scel.evaluateImpact(before, after);

      if (ok) {
        this.scel.installBlueprint(blueprint, this.cognitiveModules);
        console.log(`[SCEL] '${type}' module approved and installed.`);
      } else {
        // Rollback changes if improvement insufficient
        this.smv.clarityIndex = before.clarityIndex;
        this.smv.driftRisk = before.driftRisk;
        this.smv.consolidationTension = before.consolidationTension;
        console.log("[SCEL] Expansion rejected (insufficient improvement).");
      }
    }

    console.log(
      `[PRIME-META] Meta-state: stability=${this.smv.stabilityIndex.toFixed(3)} ` +
      `clarity=${this.smv.clarityIndex.toFixed(3)} drift=${driftRisk.toFixed(3)}`
    );
  }
}

