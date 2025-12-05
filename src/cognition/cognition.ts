// src/cognition/cognition.ts

import { IntentEngine, type IntentResult } from "../intent/intent_engine.ts";
import { IntentRouter, type RouteResponse } from "../router/intent_router.ts";
import { SafetyGuard } from "../safety/safety_guard.ts";
import { StabilityMatrix } from "../stability/stability_matrix.ts";
import { SEL } from "../emotion/sel.ts";
import { MotivationEngine } from "./motivation_engine.ts";
import { ProtoGoalEngine } from "./proto_goal_engine.ts";
import { ActionSelectionEngine } from "./action_selection_engine.ts";
import { PlanningEngine } from "./planning_engine.ts";
import { GreetingHandler } from "../handlers/greeting.ts";
import { SystemStatusHandler } from "../handlers/system_status.ts";
import { UnknownHandler } from "../handlers/unknown.ts";
import { Recall } from "./recall_engine.ts";
import type { CognitiveState } from "../shared/types.ts";
import { NeuralContextEncoder } from "../neural/context_encoder.ts";
import { NeuralSymbolicCoherence } from "./neural_symbolic_coherence.ts";
import { NeuralCausality } from "./neural_causality_engine.ts";
import { NeuralEventSegmentation } from "./neural_event_segmentation.ts";
import { UncertaintyEngine } from "./uncertainty/uncertainty_engine.ts";
import { RiskMitigationEngine } from "./risk/risk_mitigation_engine.ts";
import { ReflectiveSafetyLoop } from "../safety/reflective_safety_loop.ts";
import { DualMindConflictResolver } from "./self/conflict_resolver.ts";
import { DualIntentCoherenceEngine } from "./intent/dual_intent_coherence_engine.ts";
import { NeuralBridge } from "../neural/neural_bridge.ts";
import { DualMemoryCoherenceEngine } from "../memory/coherence/dual_memory_coherence_engine.ts";
import { integrateSharedMemory } from "../memory/memory_router.ts";
import MemoryBroker from "../memory/index.ts";

export class Cognition {
  private intentEngine = new IntentEngine();
  private router = new IntentRouter();
  private uncertainty = new UncertaintyEngine();
  private risk = new RiskMitigationEngine();
  private safetyLoop = new ReflectiveSafetyLoop();
  private dualResolver = new DualMindConflictResolver();
  private dualIntent = new DualIntentCoherenceEngine();
  private dualMemory = new DualMemoryCoherenceEngine();
  private memoryBroker = new MemoryBroker();

  constructor() {
    // Register handlers
    this.router.register("greeting", GreetingHandler);
    this.router.register("ask_status", SystemStatusHandler);
    this.router.register("unknown", UnknownHandler);
  }

  async cycle(input: string): Promise<RouteResponse> {
    // Safety check before processing
    if (!SafetyGuard.preCognitionCheck()) {
      console.warn("[PRIME-COGNITION] Safety check failed, skipping intent classification");
      return {
        type: "fallback",
        payload: { message: "Safety check failed, unable to process." }
      };
    }

    // Classify intent
    const intent = this.intentEngine.classify(input);
    console.log("[PRIME-INTENT]", intent);

    // A39: Track last intent for motivation computation
    MotivationEngine.setLastIntent(intent);

    // A38: Apply SEL influence from gravitation
    const gravity = (intent as any).gravity || 0;
    SEL.applyIntentInfluence(gravity);

    // A39: Compute motivation vector each cognition cycle
    const motivation = MotivationEngine.compute();
    console.log("[PRIME-MOTIVATION]", motivation);

    // A40: Compute proto-goals
    const goals = ProtoGoalEngine.computeGoals();
    console.log("[PRIME-GOALS]", goals);

    // A74/A79: Create cognitive state and perform recall before planning
    // A113: Initialize neural signals array
    const topGoal = goals[0] || null;
    const cognitiveState: any = {
      activeGoal: topGoal ? { type: topGoal.type } : undefined,
      confidence: intent.confidence,
      uncertainty: 1 - intent.confidence,
      motivation: motivation, // A79: Include motivation for foresight predictions
      neuralSignals: [] // A113: Initialize neural signals array
    };

    // A74: Perform preliminary recall lookup for planning
    if (topGoal) {
      // Build a simple embedding for recall lookup
      const ncel = new NeuralContextEncoder();
      const baseFeatures = [
        motivation.urgency,
        motivation.curiosity,
        motivation.claritySeeking,
        motivation.consolidation,
        intent.confidence,
        1 - intent.confidence
      ];
      // Create a simple lookup vector (pad to 64 for similarity search)
      const lookupVector = new Array(64).fill(0);
      baseFeatures.forEach((v, i) => {
        if (i < lookupVector.length) {
          lookupVector[i] = Math.max(-1, Math.min(1, v));
        }
      });

      const recallResults = Recall.recall(lookupVector, 3);
      const recallSummary = Recall.summarizeRecall(recallResults);
      if (recallSummary) {
        cognitiveState.recall = recallSummary;
      }
    }

    // A124: Compute uncertainty vector and score
    cognitiveState.noiseLevel = cognitiveState.noiseLevel ?? 0.1;
    cognitiveState.predictionVariance = cognitiveState.predictionVariance ?? 0;
    cognitiveState.conceptDrift = cognitiveState.conceptDrift ?? 0;
    cognitiveState.ambiguity = cognitiveState.ambiguity ?? 0;
    cognitiveState.relationalUncertainty = cognitiveState.relationalUncertainty ?? 0;

    cognitiveState.uncertaintyVector = this.uncertainty.compute({
      noise: cognitiveState.noiseLevel,
      variance: cognitiveState.predictionVariance,
      conceptDrift: cognitiveState.conceptDrift,
      situationalAmbiguity: cognitiveState.ambiguity,
      trustFluctuation: cognitiveState.relationalUncertainty
    });

    cognitiveState.uncertaintyScore = this.uncertainty.summarize(cognitiveState.uncertaintyVector);

    console.log("[PRIME-UNCERTAINTY]", {
      vector: cognitiveState.uncertaintyVector,
      score: cognitiveState.uncertaintyScore.toFixed(3)
    });

    // A125: Assess cognitive risk and choose mitigation strategy
    cognitiveState.narrativeInstability = cognitiveState.narrativeInstability ?? 0;
    cognitiveState.emotionDrift = cognitiveState.emotionDrift ?? 0;

    const riskScore = this.risk.assess({
      conceptDrift: cognitiveState.conceptDrift ?? 0,
      predictionVariance: cognitiveState.predictionVariance ?? 0,
      uncertainty: cognitiveState.uncertaintyScore ?? 0,
      narrativeInstability: cognitiveState.narrativeInstability,
      emotionalDrift: cognitiveState.emotionDrift
    });

    cognitiveState.riskScore = riskScore;
    const mitigation = this.risk.chooseMitigation(riskScore);
    cognitiveState.mitigationStrategy = mitigation;

    console.log("[PRIME-RISK]", {
      riskScore: riskScore.toFixed(3),
      mitigation: mitigation
    });

    // A126: Run reflective safety loop if stabilization is needed
    if (mitigation === "stabilize_and_pause") {
      console.log("[PRIME-SAFETY] Running reflective safety loop...");
      
      // Capture state before repair
      const beforeFactors = this.safetyLoop.analyzeInstability(cognitiveState);
      
      // Run the safety loop (applies repairs)
      const logs = this.safetyLoop.run(cognitiveState);
      console.log("[PRIME-SAFETY-LOOP]", logs);
      
      // Check if stability improved after repair
      const afterFactors = this.safetyLoop.analyzeInstability(cognitiveState);
      const improved = this.safetyLoop.checkStabilityImprovement(beforeFactors, afterFactors);
      
      if (improved) {
        console.log("[PRIME-KERNEL] Stability improving. Resuming normal cognition.");
      } else {
        console.log("[PRIME-KERNEL] Stability check: monitoring ongoing...");
      }
    }

    // A126: Handle critical halt scenario
    if (mitigation === "halt_and_request_operator") {
      console.log("[PRIME-SAFETY] Mitigation: halt_and_request_operator");
      console.log("[PRIME-SAFETY-LOOP] PRIME awaiting your guidance.");
    }

    // A127: Resolve conflicts with SAGE if dual-mind is active
    const sageState = (globalThis as any).__SAGE_STATE__ || null;
    if (sageState) {
      const outcome = this.dualResolver.resolve(cognitiveState, sageState);
      console.log("[PRIME-DUAL-CONFLICT]", {
        resolved: outcome.resolved,
        strategy: outcome.strategy,
        severity: outcome.severity?.toFixed(3),
        conflict: outcome.conflict
      });

      if (outcome.strategy === "halt_and_request_operator") {
        console.log("[PRIME-DUAL] Operator input required.");
        // Don't halt the entire cognition cycle, but mark that operator input is needed
        cognitiveState.awaitingOperator = true;
      }

      // A128: Compute unified intent from both PRIME and SAGE
      const dualIntent = this.dualIntent.computeCoherentIntent(cognitiveState, sageState);
      cognitiveState.unifiedIntent = dualIntent;
      console.log("[PRIME↔SAGE INTENT]", {
        unifiedGoal: dualIntent.unifiedGoal,
        unifiedUrgency: dualIntent.unifiedUrgency.toFixed(3),
        unifiedPredictionVector: dualIntent.unifiedPredictionVector?.slice(0, 3),
        weights: {
          prime: dualIntent.weights.prime.toFixed(3),
          sage: dualIntent.weights.sage.toFixed(3)
        }
      });
      console.log("[DUAL-MIND] Unified intent generated.");

      // A128: Share unified intent with SAGE via neural bridge
      NeuralBridge.sendUnifiedIntent(dualIntent);

      // A129: Process shared memory coherence with SAGE
      const sageMemory = sageState.memory ?? null;
      if (sageMemory) {
        const primeMemoryState = this.memoryBroker.exportState();
        const packet = this.dualMemory.generateCoherencePacket(primeMemoryState, sageMemory);
        
        // Integrate shared memory into PRIME
        integrateSharedMemory(packet.mergedMemory);
        
        console.log("[PRIME↔SAGE MEMORY]", {
          drift: packet.drift.toFixed(3),
          mergedEpisodes: packet.mergedMemory.episodic.length
        });

        // A129: Send shared memory packet back to SAGE
        if (sageState.receiveSharedMemory) {
          sageState.receiveSharedMemory(packet);
        } else {
          // Fallback: store in global state
          (globalThis as any).__SHARED_MEMORY_PACKET__ = packet;
        }
        console.log("[DUAL-MIND] Shared memory packet sent to SAGE.");
      }
    }

    // A42/A75: Generate plans for each goal (with recall-informed and concept-informed cognitive state)
    // A75: Match concept before planning if embedding is available
    if (cognitiveState.recall && cognitiveState.recall.reference) {
      // If we have recall, we likely have an embedding in the neural context
      // For now, concepts will be matched during reflection, but we could do it here too
    }
    
    const plans = goals.map((g) => PlanningEngine.generatePlan(g, cognitiveState));
    console.log("[PRIME-PLANS]", plans);

    // A42: Choose best plan by score
    const best = plans.sort((a, b) => b.score - a.score)[0] || null;
    if (best && best.steps.length > 0) {
      console.log("[PRIME-PLAN] Selected plan:", {
        goal: best.goal.type,
        score: best.score,
        steps: best.steps.map((s) => s.name),
        counterfactual: best.counterfactual
      });
      // Execute plan steps one-by-one
      for (const step of best.steps) {
        console.log("[PRIME-PLAN-STEP] Executing:", step.name);
        try {
          step.fn();
        } catch (err) {
          console.error("[PRIME-PLAN-ERROR]", err);
          break;
        }
      }
    }

    // Update stability metrics
    StabilityMatrix.update("cognition", {
      latency: 0, // Intent classification is fast
      load: intent.confidence,
      errors: intent.intent === "unknown" ? 1 : 0,
    });

    // Route to appropriate handler
    const routed = await this.router.route(intent);
    console.log("[PRIME-ROUTE]", routed);

    // A35: Post-cognition normalization
    this.finalizeCognition(routed, intent);

    // A116: Update Joint Situation Modeler with PRIME context
    const JSM = (globalThis as any).__PRIME_JSM__;
    if (JSM) {
      const primeContext = {
        timestamp: Date.now(),
        activeGoal: cognitiveState.activeGoal,
        recommendedFocus: cognitiveState.activeGoal?.type,
        coherenceScore: intent.confidence,
        stability: intent.confidence,
        salience: [], // Can be populated from situation model
        summary: `Processing intent: ${intent.intent}`
      };
      
      // Get SAGE state from federation (if available)
      const sageState = (globalThis as any).__SAGE_STATE__ || null;
      JSM.update(primeContext, sageState);
    }

    // A115: Detect event boundaries
    const eventSegEngine = (globalThis as any).__PRIME_EVENT_SEG__;
    if (eventSegEngine) {
      const state = {
        intent: intent.intent,
        emotion: SEL.getState(),
        goal: cognitiveState.activeGoal?.type,
        embedding: cognitiveState.embedding,
        stability: intent.confidence,
        predictionError: 0 // Can be computed from prediction engine
      };

      const boundary = eventSegEngine.detectEventBoundary(state);
      if (boundary) {
        // Capture boundary in episodic archive
        const episodicArchive = (globalThis as any).__PRIME_EPISODIC_ARCHIVE__;
        if (episodicArchive && episodicArchive.captureBoundary) {
          episodicArchive.captureBoundary(boundary);
        }

        // Broadcast event boundary
        const resonanceBus = (globalThis as any).__PRIME_RESONANCE_BUS__;
        if (resonanceBus && resonanceBus.broadcast) {
          resonanceBus.broadcast("event_boundary", boundary);
        } else if (resonanceBus && resonanceBus.publish) {
          resonanceBus.publish({ type: "event_boundary", data: boundary });
        }
      }
    }

    return routed;
  }

  finalizeCognition(result: RouteResponse, intent: IntentResult) {
    const emotion = SEL.getState();
    
    // A35: cognition improves coherence gradually
    const newCoherence = Math.min(1, emotion.coherence + 0.02);
    
    // Reduce tension after successful cognition pass
    const newTension = Math.max(0, emotion.tension - 0.03);
    
    // Update emotion state
    SEL.updateEmotion({
      stability: newCoherence,
      tensionSignal: newTension,
    });

    // A36: Determine cognition outcome
    let outcome: "success" | "confusion" | "failure" = "success";
    if (result && (result as any).qualityScore !== undefined) {
      const qualityScore = (result as any).qualityScore;
      if (qualityScore > 0.8) outcome = "success";
      else if (qualityScore > 0.4) outcome = "confusion";
      else outcome = "failure";
    } else {
      // Heuristic: if result type is "fallback" or "unknown", treat as confusion/failure
      if (result.type === "fallback") {
        outcome = "failure";
      } else if (result.type === "unknown") {
        outcome = "confusion";
      } else {
        // Default to success for known intent types
        outcome = "success";
      }
    }

    // Apply emotional reinforcement
    SEL.reinforce(outcome);

    // A38: Reinforce intent gravitation based on outcome
    IntentEngine.reinforceIntent(intent.intent, outcome);

    // Log outcome for debugging
    console.log(`[PRIME-SEL] Reinforcement outcome: ${outcome}`);
  }

  selectReasoningPath(paths: Array<{ type: string; weight: number; [key: string]: any }>) {
    const emotion = SEL.getState();

    return paths
      .map((p) => {
        let modified = p.weight;

        // Emotion-aware modulation:
        modified += (emotion.valence - 0.5) * 0.2;
        modified += (emotion.certainty - 0.5) * 0.3;

        // High tension pushes PRIME toward cautionary paths
        if (emotion.tension > 0.6 && p.type === "cautious") {
          modified += 0.3;
        }

        // Low coherence penalizes high-risk paths
        if (emotion.coherence < 0.4 && p.type === "aggressive") {
          modified -= 0.3;
        }

        return { ...p, adjustedWeight: modified };
      })
      .sort((a, b) => b.adjustedWeight - a.adjustedWeight)[0];
  }
}

export const cognition = new Cognition();

