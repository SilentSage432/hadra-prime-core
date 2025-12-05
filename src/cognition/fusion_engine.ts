// src/cognition/fusion_engine.ts
// A89: Neural Inference Boundary Controller Integration
// A90: Neural Interaction Contract Integration

import type { ToneVector } from "../expression/tone/tone_detector.ts";
import type { CognitiveState } from "./cognitive_state.ts";
import { MemoryStore } from "../memory/memory_store.ts";
import { CognitiveStabilizer } from "./stability/cognitive_stabilizer.ts";
import { ReasoningIntegrity } from "./bias/integrity_layer.ts";
import { TemporalEngine } from "./temporal_engine.ts";
import { PredictiveCoherence } from "./predictive_coherence.ts";
import { ClusterBus } from "../distributed/cluster_bus.ts";
import { PredictiveConsensus } from "../distributed/predictive_consensus.ts";
import { SEL } from "../emotion/sel.ts";
import type { IntentResult } from "../intent/intent_engine.ts";
import { NeuralBoundary } from "../neural/boundary_controller.ts";
import { NeuralContextEncoder } from "../neural/context_encoder.ts";
import { MotivationEngine } from "./motivation_engine.ts";
import { NIC, type NeuralInputContract } from "../neural/contract/neural_interaction_contract.ts";
import type { SymbolicPacket } from "../shared/symbolic_packet.ts";

export class FusionEngine {
  private memory: MemoryStore;

  constructor(memory: MemoryStore) {
    this.memory = memory;
  }

  async buildCognitiveState(intent: any, tone: ToneVector, contextSnapshot: any): Promise<CognitiveState> {
    const start = performance.now();

    // Attach temporal context to payload before fusion
    const payload = { intent, tone, contextSnapshot };
    const payloadWithTemporal = TemporalEngine.attachTemporalContext(payload);
    
    // Inject consensus prediction into the fusion payload (uses combined intelligence of all threads)
    // FIXED: Only compute consensus when explicitly requested to prevent recursion storms
    // Predictions should be event-driven, not automatic
    if ((payload as any).allowPrediction) {
      payload.prediction = PredictiveCoherence.computeConsensus();
    } else {
      payload.prediction = {
        horizon: "idle" as const,
        stabilityTrend: "stable" as const,
        likelyNextIntent: null,
        recursionRisk: 0,
      };
    }
    
    // Merge temporal context and prediction into contextSnapshot for downstream use
    const enrichedContext = {
      ...contextSnapshot,
      temporal: payloadWithTemporal.temporal,
      prediction: payload.prediction,
    };

    const memoryRecall = this.memory.retrieveRelevant(intent?.type || "general");

    // A46: Extract multimodal events from recent events
    const recentEvents = enrichedContext.recentEvents || [];
    const multimodal = {
      vision: recentEvents.filter((e: any) => e.type === "vision"),
      audio: recentEvents.filter((e: any) => e.type === "audio"),
      symbolic: recentEvents.filter((e: any) => e.type === "symbolic"),
    };

    const priorityLevel = this.computePriority(intent, enrichedContext);
    const riskLevel = this.computeRisk(intent);
    const operatorFocus = this.inferOperatorFocus(intent, enrichedContext, tone);
    const recommendedResponseMode = this.determineResponseMode(
      operatorFocus,
      tone,
      priorityLevel,
      riskLevel
    );

    // Build fusion payload
    const fusion = {
      meaning: {
        priorityLevel,
        riskLevel,
        operatorFocus,
        recommendedResponseMode,
        memoryRecall,
      },
      intent,
      context: enrichedContext,
      stabilityScore: 0.6, // Default stability score
    };

    // Emotion-influenced fusion weighting
    this.applyEmotionToFusion(fusion);

    // Measure fusion latency
    const latency = performance.now() - start;

    // Stabilize the fusion
    const stabilized = CognitiveStabilizer.stabilize(fusion, latency);

    // If stabilization returned null or degraded, use simplified state
    if (!stabilized || stabilized.meaning?.degraded) {
      return {
        intent: stabilized?.intent || intent,
        tone,
        context: stabilized?.context || enrichedContext,
        memory: memoryRecall,
        priorityLevel: stabilized?.meaning?.priorityLevel || priorityLevel,
        riskLevel: stabilized?.meaning?.riskLevel || riskLevel,
        operatorFocus: stabilized?.meaning?.operatorFocus || operatorFocus,
        recommendedResponseMode: stabilized?.meaning?.recommendedResponseMode || recommendedResponseMode,
      };
    }

    // Evaluate reasoning integrity and correct bias
    const integrityResult = ReasoningIntegrity.evaluate(stabilized);
    const corrected = ReasoningIntegrity.correct(stabilized, integrityResult.integrity);

    // Store integrity result for downstream use
    (corrected as any).integrity = integrityResult;

    // Return stabilized and integrity-corrected cognitive state
    const cognitiveState: CognitiveState = {
      intent: corrected.intent,
      tone,
      context: corrected.context,
      memory: corrected.meaning.memoryRecall,
      priorityLevel: corrected.meaning.priorityLevel,
      riskLevel: corrected.meaning.riskLevel,
      operatorFocus: corrected.meaning.operatorFocus,
      recommendedResponseMode: corrected.meaning.recommendedResponseMode,
    };

    // Attach integrity result for downstream use
    (cognitiveState as any).integrity = integrityResult;

    // A46: Attach multimodal data to cognitive state
    (cognitiveState as any).multimodal = multimodal;

    // A89/A90: Neural assistance pathway (if allowed)
    const motivations = MotivationEngine.compute();
    if (NeuralBoundary.allowNeuralInference(motivations)) {
      console.log("[NEURAL] Neural inference permitted.");
      
      const stabilitySnapshot = {
        score: (fusion as any).stabilityScore || 0.6,
        stabilityScore: (fusion as any).stabilityScore || 0.6,
        recursionRisk: 0,
        coherenceScore: SEL.getState().coherence
      };
      
      const embedding = new NeuralContextEncoder().encodeContext(
        cognitiveState,
        motivations,
        stabilitySnapshot
      );
      
      const safeEmbedding = NeuralBoundary.sanitizeInput(embedding.vector);
      
      // A90: Build Neural Input Contract
      const recentEvents = (enrichedContext.recentEvents || []).slice(0, 5).map((e: any) => {
        return typeof e === "string" ? e : JSON.stringify(e);
      });
      
      const goalContext = intent?.type || cognitiveState.intent?.type || null;
      
      const neuralInput: NeuralInputContract = {
        embedding: safeEmbedding,
        motivations: {
          curiosity: motivations.curiosity,
          claritySeeking: motivations.claritySeeking,
          consolidation: motivations.consolidation,
          stabilityPressure: motivations.stabilityPressure
        },
        recentEvents: recentEvents,
        goalContext: goalContext,
        timestamp: Date.now()
      };
      
      // A90: Validate input contract
      if (!NIC.validateInput(neuralInput)) {
        console.log("[NEURAL] Input contract invalid. Blocking neural request.");
      } else {
        // FUTURE HOOK: PyTorch inference call
        // const rawOutput = await pytorchModel.infer(neuralInput);
        const rawOutput = null; // placeholder for A90–A95
        
        // A90: Validate output contract
        if (!rawOutput || !NIC.validateOutput(rawOutput)) {
          console.log("[NEURAL] Output contract invalid. Rejecting neural inference.");
        } else {
          const validated = NeuralBoundary.validateNeuralOutput(rawOutput);
          if (validated && NeuralBoundary.approve(validated, motivations)) {
            (cognitiveState as any).neuralAssistance = validated;
            console.log(`[NEURAL] Assistance approved: ${JSON.stringify(validated)}`);
          } else {
            console.log("[NEURAL] Assistance rejected or unavailable.");
          }
        }
      }
    }

    // A101: Neural Bridge Integration - Cross-Modal Embedding Bridge
    if ((globalThis as any).PRIME_NEURAL_BRIDGE) {
      try {
        const packet: SymbolicPacket = {
          type: "cognitive_state",
          payload: {
            intent: cognitiveState.intent,
            priorityLevel: cognitiveState.priorityLevel,
            riskLevel: cognitiveState.riskLevel,
            operatorFocus: cognitiveState.operatorFocus
          }
        };

        const encoded = (globalThis as any).PRIME_NEURAL_BRIDGE.encodeToTensor(packet);
        const neuralOut = await (globalThis as any).PRIME_NEURAL_BRIDGE.runNeuralModel(encoded);
        const decoded = (globalThis as any).PRIME_NEURAL_BRIDGE.decodeFromTensor(neuralOut);
        
        // Attach neural enhancement to cognitive state
        (cognitiveState as any).neural_enhancement = decoded;
      } catch (error) {
        console.warn("[PRIME-NEURAL] Neural bridge processing failed:", error);
      }
    }

    // Synthetic Emotion Integration
    this.finalizeFusion(cognitiveState);

    return cognitiveState;
  }

  // A38: Compute consensus from multiple intents using gravitation
  static computeConsensus(intents: IntentResult[]): IntentResult | null {
    if (!intents.length) return null;

    // Apply gravitation boost to each intent
    const weighted = intents.map((i) => {
      const g = (i as any).gravity || 0;
      return { ...i, weight: (i.confidence || 0.5) + g };
    });

    // Sort by combined confidence + gravity
    weighted.sort((a, b) => (b.weight || 0) - (a.weight || 0));

    // WINNER: the intent PRIME is "pulled" toward
    const winner = weighted[0];
    return winner;
  }

  private applyEmotionToFusion(fusion: any) {
    const emotion = SEL.getState();

    // Modify fusion weighting based on coherence / tension:
    const coherenceBoost = emotion.coherence * 0.15;
    const tensionPenalty = emotion.tension * 0.25;

    fusion.stabilityScore = Math.max(
      0,
      Math.min(
        1,
        (fusion.stabilityScore ?? 0.6)
        + coherenceBoost
        - tensionPenalty
      )
    );
  }

  private finalizeFusion(fusion: any) {
    // Synthetic Emotion Integration
    SEL.updateEmotion({
      stability: fusion.stabilityScore ?? 0.8,
      certainty: fusion.confidence ?? 0.5,
      tensionSignal: fusion.contradictionLevel ?? 0.0,
    });
  }

  private computePriority(intent: any, context: any): number {
    if (!intent) return 0.2;
    if (intent.type === "error") return 1.0;
    if (intent.type === "system_action") return 0.9;
    if (intent.type === "question") return 0.5;
    return 0.3;
  }

  private computeRisk(intent: any): number {
    if (!intent) return 0.1;
    if (intent.type === "system_action") return 0.7;
    if (intent.type === "dangerous") return 1.0;
    return 0.1;
  }

  private inferOperatorFocus(intent: any, context: any, tone: ToneVector): "build" | "debug" | "ideate" | "learn" | "unknown" {
    if (tone.emotionalState === "frustrated") return "debug";

    const lastIntent = context?.latestSession?.value || {};

    if (lastIntent.type === "build_step") return "build";
    if (lastIntent.type === "phase_switch") return "ideate";
    if (intent?.type === "question") return "learn";

    return "unknown";
  }

  private determineResponseMode(
    operatorFocus: string,
    tone: ToneVector,
    priority: number,
    risk: number
  ): "direct" | "supportive" | "detailed" | "cautious" {
    if (risk > 0.7) return "cautious";
    if (operatorFocus === "debug") return "direct";
    if (operatorFocus === "learn") return "detailed";
    if (tone.emotionalState === "frustrated") return "supportive";
    return "direct";
  }
}

// FIXED: ClusterBus listener disabled to prevent recursion storms
// Consensus computation should be event-driven, not automatic
// ClusterBus.onSnapshot((snapshot) => {
//   PredictiveConsensus.registerSnapshot(snapshot);
//   const result = PredictiveConsensus.computeConsensus();
//   console.log(
//     `[PRIME-DIST][Consensus] agreement=${result.agreement.toFixed(2)} samples=${result.samples}`
//   );
// });

