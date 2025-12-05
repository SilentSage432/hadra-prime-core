// src/cognition/reflection_engine.ts
// A74: Neural Recall Integration
// A75: Concept Formation Integration
// A76: Hierarchical Concept Networks
// A77: Knowledge Graph Integration
// A78: Inference Engine Integration
// A79: Long-Range Predictive Model
// A81: Meta-Self Awareness Engine Integration
// A82: Temporal Identity Engine Integration
// A83: Proto-Narrative Engine Integration
// A84: Internal Dialogue Engine Integration
// A85: Multi-Voice Deliberation Engine Integration
// A86: Internal Conflict Resolution Engine Integration
// A87: Cognitive Realignment Engine Integration
// A88: Cognitive Homeostasis System Integration
// A93: Neural Memory Encoding Integration
// A95: Concept Formation Engine Integration

import { Concepts } from "./concepts/concept_engine.ts";
import { Hierarchy } from "./concepts/concept_hierarchy.ts";
import { Knowledge } from "./knowledge/knowledge_graph.ts";
import { Inference } from "./inference/inference_engine.ts";
import { Foresight } from "./prediction/foresight_engine.ts";
import { MetaSelf } from "./self/meta_self_engine.ts";
import { TemporalIdentity } from "./self/temporal_identity_engine.ts";
import { Narrative } from "./self/narrative_engine.ts";
import { InternalDialogue } from "./self/internal_dialogue_engine.ts";
import { MultiVoices } from "./self/multivoice_engine.ts";
import { ConflictEngine } from "./self/conflict_resolver.ts";
import { Realignment } from "./self/realignment_engine.ts";
import { Homeostasis } from "./self/homeostasis_engine.ts";
import { recallSimilar } from "../memory/memory_router.ts";
import { ConceptGraph } from "../memory/concepts/concept_store.ts";

export class ReflectionEngine {
  // A93: Neural recall integration method
  async integrateNeuralRecall(embedding: number[]) {
    const similar = await recallSimilar(embedding, 3);
    return similar;
  }

  reflect(cognitiveState: any, selState: any) {
    const reflection = {
      timestamp: Date.now(),
      motivation: cognitiveState.motivation,
      topGoal: cognitiveState.topGoal,
      sel: selState,
      summary: this.generateSummary(cognitiveState, selState)
    };

    // A74: Log recall information if available
    if (cognitiveState.recall?.reference) {
      this.logReflection(
        `Recalling prior experience: ${JSON.stringify(cognitiveState.recall.reference)}`
      );
      console.log("[PRIME-RECALL] Reflection informed by past experience:", {
        intuition: cognitiveState.recall.intuition.toFixed(3),
        reference: cognitiveState.recall.reference
      });
    }

    // A75: Match current embedding to concepts
    // A93: Integrate neural recall if embedding is available
    if (cognitiveState.embedding && cognitiveState.embedding.length > 0) {
      // A93: Perform neural recall
      this.integrateNeuralRecall(cognitiveState.embedding).then((neuralHints) => {
        if (neuralHints.length > 0) {
          console.log("[PRIME-REFLECTION] Neural recall hints:", neuralHints.map(h => ({
            id: h.id,
            score: h.score.toFixed(3),
            tags: h.tags
          })));
        }
      }).catch(err => {
        console.warn("[PRIME-REFLECTION] Neural recall error:", err);
      });

      const concept = Concepts.matchConcept(cognitiveState.embedding);
      if (concept && concept.strength >= 3) {
        this.logReflection(
          `Recognized emerging concept: ${concept.id} (strength=${concept.strength})`
        );
        console.log("[PRIME-CONCEPT] Reflection matched to concept:", {
          conceptId: concept.id,
          strength: concept.strength,
          members: concept.members.length
        });
        // Attach concept to cognitive state
        (cognitiveState as any).concept = concept;

        // A76: Find domain that contains this concept
        const domain = Hierarchy.findDomainForConcept(concept.id);
        if (domain) {
          this.logReflection(
            `Recognized domain influence: ${domain.id} (strength=${domain.strength})`
          );
          console.log("[PRIME-DOMAIN] Reflection influenced by domain:", {
            domainId: domain.id,
            strength: domain.strength,
            concepts: domain.concepts.length,
            metaConcepts: domain.metaConcepts.length
          });
          // Attach domain to cognitive state
          (cognitiveState as any).domain = domain;
          // Also attach to reflection object so it can be accessed by kernel
          (reflection as any).domain = domain;
        }
      }
    }

    console.log("[PRIME-REFLECTION]", reflection.summary);

    // A77: Add reflection to knowledge graph
    const reflectionId = `reflection_${reflection.timestamp}`;
    Knowledge.addNode(reflectionId, "reflection", {
      summary: reflection.summary,
      state: {
        motivation: reflection.motivation,
        topGoal: reflection.topGoal,
        sel: reflection.sel
      },
      timestamp: reflection.timestamp,
    }, 1.0);

    // A77: Link reflection to related concepts and domains
    if (cognitiveState.concept) {
      Knowledge.linkContainment(cognitiveState.concept.id, reflectionId);
    }
    if (cognitiveState.domain) {
      Knowledge.linkContainment(cognitiveState.domain.id, reflectionId);
    }
    if (cognitiveState.embedding) {
      // Link to related memories via embedding similarity
      // (This will be handled by the memory bank's similarity links)
    }

    // A78: Perform inference operations
    if (cognitiveState.embedding) {
      // Infer latent graph links
      const latent = Inference.deriveLatentLinks();
      if (latent.length > 0) {
        this.logReflection(`Latent conceptual relationships detected: ${latent.length}`);
        console.log("[PRIME-INFERENCE] Latent links derived:", latent.length);
      }

      // Infer active domain
      const domainGuess = Inference.inferActiveDomain(cognitiveState.embedding);
      if (domainGuess.domain && domainGuess.confidence > 0.5) {
        this.logReflection(
          `Inference: Current state aligns with domain ${domainGuess.domain.id} (conf=${domainGuess.confidence.toFixed(2)})`
        );
        console.log("[PRIME-INFERENCE] Active domain inferred:", {
          domainId: domainGuess.domain.id,
          confidence: domainGuess.confidence.toFixed(3)
        });
      }

      // Suggest strategy
      const strategy = Inference.inferStrategy(cognitiveState.embedding);
      if (strategy && strategy.confidence > 0.3) {
        this.logReflection(
          `Inference: Recommended strategy — ${strategy.strategy} (conf=${strategy.confidence.toFixed(2)})`
        );
        console.log("[PRIME-INFERENCE] Strategy inferred:", {
          strategy: strategy.strategy,
          confidence: strategy.confidence.toFixed(3)
        });
        // Attach strategy to reflection for planning use
        (reflection as any).inferredStrategy = strategy;
      }
    }

    // A79: Perform foresight predictions
    if (cognitiveState.motivation && selState) {
      try {
        const foresight = Foresight.forecastSystemState(
          cognitiveState.motivation,
          selState
        );

        // Log motivation projections
        if (foresight.projectedMotivation.length > 0) {
          const finalProjection = foresight.projectedMotivation[
            foresight.projectedMotivation.length - 1
          ];
          
          this.logReflection(
            `Foresight: consolidation expected to decrease to ${finalProjection.consolidation.toFixed(3)}`
          );
          
          this.logReflection(
            `Foresight: claritySeeking projected to rise toward ${finalProjection.claritySeeking.toFixed(3)}`
          );

          this.logReflection(
            `Foresight: curiosity projected to ${finalProjection.curiosity.toFixed(3)}`
          );

          console.log("[PRIME-FORESIGHT] Motivation trajectory projected:", {
            steps: foresight.projectedMotivation.length,
            finalConsolidation: finalProjection.consolidation.toFixed(3),
            finalClaritySeeking: finalProjection.claritySeeking.toFixed(3),
            finalCuriosity: finalProjection.curiosity.toFixed(3)
          });
        }

        // Log SEL projections
        if (foresight.projectedSEL.length > 0) {
          const finalSEL = foresight.projectedSEL[foresight.projectedSEL.length - 1];
          
          this.logReflection(
            `Foresight: SEL stability pressure trajectory ${(finalSEL.stabilityPressure || 0).toFixed(4)}`
          );

          this.logReflection(
            `Foresight: coherence projected to ${finalSEL.coherence.toFixed(3)}`
          );

          console.log("[PRIME-FORESIGHT] SEL trajectory projected:", {
            steps: foresight.projectedSEL.length,
            finalCoherence: finalSEL.coherence.toFixed(3),
            finalTension: finalSEL.tension.toFixed(3),
            finalStabilityPressure: (finalSEL.stabilityPressure || 0).toFixed(4)
          });
        }

        // Log domain forecasts
        if (foresight.domainForecast.length > 0) {
          console.log("[PRIME-FORESIGHT] Domain trends projected:", 
            foresight.domainForecast.map(d => ({
              domainId: d.id,
              projectedStrength: d.projectedStrength.toFixed(3)
            }))
          );
        }

        // Attach foresight to reflection for planning use
        (reflection as any).foresight = foresight;
      } catch (error) {
        console.warn("[PRIME-FORESIGHT] Error during prediction:", error);
      }
    }

    // A81: Update Meta-Self Awareness Engine during reflection
    if (cognitiveState.motivation && selState) {
      const state = cognitiveState.motivation;
      
      // Update internal state with motivation values
      MetaSelf.updateInternalState("motivation.consolidation", state.consolidation || 0);
      MetaSelf.updateInternalState("motivation.curiosity", state.curiosity || 0);
      MetaSelf.updateInternalState("motivation.claritySeeking", state.claritySeeking || 0);
      
      // Update cognitive capabilities
      // Estimate reflection depth from SEL coherence and tension
      const reflectionDepth = (selState.coherence || 0) * (1 - (selState.tension || 0));
      MetaSelf.updateCapability("reflectionDepth", reflectionDepth);
      
      // Estimate prediction depth from foresight availability
      const predictionDepth = (reflection as any).foresight ? 0.7 : 0.3;
      MetaSelf.updateCapability("predictionDepth", predictionDepth);
      
      // Update emotional profile from SEL state
      // Map SEL coherence to stability, curiosity to exploration, consolidation to consolidation
      const stability = selState.coherence || 0.5;
      const exploration = state.curiosity || 0.5;
      const consolidation = state.consolidation || 0.5;
      MetaSelf.updateEmotionalProfile(stability, exploration, consolidation);
      
      // Adjust growth trajectory
      MetaSelf.adjustGrowth("selfReflection", 0.001);
      
      // Compute and update stability score
      const stabilityScore = MetaSelf.computeStabilityScore();
      
      // Update last reflection timestamp
      const selfModel = MetaSelf.exportModel();
      selfModel.lastReflection = reflection.timestamp;
      
      this.logReflection(
        `[SELF] Updated internal self-model (stability=${stabilityScore.toFixed(3)})`
      );

      // A82: Take temporal snapshot of current self
      TemporalIdentity.takeSnapshot();

      // A82: Generate future self projection
      const futureSelf = TemporalIdentity.computeFutureSelf();
      this.logReflection(
        `[SELF-TIME] Projection: stability=${futureSelf.predictedStability.toFixed(3)}, reasoningDepth=${futureSelf.expectedGrowth.reasoningDepth.toFixed(3)}`
      );

      // A83: Create narrative entry for internal storyline
      const topGoal = cognitiveState.topGoal;
      const emotionalBias = {
        claritySeeking: (state.claritySeeking || 0).toFixed(3),
        exploration: (state.curiosity || 0).toFixed(3),
        consolidation: (state.consolidation || 0).toFixed(3)
      };
      
      const emotionalBiasStr = `{ claritySeeking: ${emotionalBias.claritySeeking}, exploration: ${emotionalBias.exploration}, consolidation: ${emotionalBias.consolidation} }`;
      
      const narrativeEntry = Narrative.createEntry({
        focus: topGoal?.type || "none",
        motivation: topGoal?.priority?.toFixed(3) || "0.000",
        emotionalBias: emotionalBiasStr,
        interpretation: `PRIME shifted attention toward '${topGoal?.type || "none"}' based on internal pressure dynamics.`
      });
      
      this.logReflection(narrativeEntry);

      // A84: Generate internal dialogue turn
      const dialogueTurn = InternalDialogue.generateDialogueTurn({
        focus: topGoal?.type || "none",
        reason: topGoal?.reason || "internal pressure dynamics",
        claritySeeking: state.claritySeeking || 0
      });
      
      this.logReflection(dialogueTurn);

      // A85: Multi-voice deliberation
      const uncertainty = 1 - (selState.certainty || 0);
      const deliberation = MultiVoices.deliberate({
        goal: topGoal?.type || "none",
        clarity: state.claritySeeking || 0,
        curiosity: state.curiosity || 0,
        uncertainty: uncertainty,
        stabilityPressure: state.stabilityPressure || 0
      });
      
      this.logReflection(deliberation.conclusion);

      // A86: Resolve internal conflicts
      const conflict = ConflictEngine.resolveConflict(
        deliberation.voices,
        state
      );
      
      this.logReflection(
        `[CONFLICT] dissonance=${conflict.dissonance.toFixed(3)} → ${conflict.result}\nstrategy=${conflict.strategy}`
      );

      // A87: Apply realignment strategy
      const realignedMotivations = Realignment.applyStrategy(
        conflict.strategy,
        state
      );
      
      // Update the motivation state with realigned values
      Object.assign(state, realignedMotivations);
      
      // Update MetaSelf with realigned values
      if (realignedMotivations.claritySeeking !== undefined) {
        MetaSelf.updateInternalState("motivation.claritySeeking", realignedMotivations.claritySeeking);
      }
      if (realignedMotivations.curiosity !== undefined) {
        MetaSelf.updateInternalState("motivation.curiosity", realignedMotivations.curiosity);
      }
      if (realignedMotivations.stabilityPressure !== undefined) {
        MetaSelf.updateInternalState("motivation.stabilityPressure", realignedMotivations.stabilityPressure);
      }
      
      this.logReflection(
        `[REALIGNMENT] Updated motivations: ${JSON.stringify(realignedMotivations)}`
      );

      // A88: Homeostasis balance regulation
      const before = { ...state };
      const regulated = Homeostasis.regulate(state);
      Object.assign(state, regulated);
      
      // Update MetaSelf with regulated values
      if (regulated.curiosity !== undefined) {
        MetaSelf.updateInternalState("motivation.curiosity", regulated.curiosity);
      }
      if (regulated.claritySeeking !== undefined) {
        MetaSelf.updateInternalState("motivation.claritySeeking", regulated.claritySeeking);
      }
      if (regulated.consolidation !== undefined) {
        MetaSelf.updateInternalState("motivation.consolidation", regulated.consolidation);
      }
      if (regulated.stabilityPressure !== undefined) {
        MetaSelf.updateInternalState("motivation.stabilityPressure", regulated.stabilityPressure);
      }
      
      this.logReflection(
        `[HOMEOSTASIS] Adjusted motivations:\n` +
        `before=${JSON.stringify(before)}\n` +
        `after=${JSON.stringify(regulated)}`
      );
    }

    // A95: Show concepts if available in cognitive state or memory
    if (cognitiveState.concept) {
      const conceptId = cognitiveState.concept;
      const concept = ConceptGraph.find(c => c.id === conceptId);
      if (concept) {
        console.log("[PRIME-REFLECTION] Concept:", {
          id: concept.id,
          label: concept.label,
          confidence: concept.confidence.toFixed(3),
          stability: concept.stability.toFixed(3),
          members: concept.members.length
        });
      }
    }

    return reflection;
  }
  
  private logReflection(message: string) {
    console.log("[PRIME-REFLECTION-RECALL]", message);
  }

  private generateSummary(cog: any, sel: any): string {
    const goal = cog.topGoal?.type || "none";
    const motivation = cog.motivation || {};
    
    return `Reflecting: PRIME prioritized '${goal}' due to ` +
           `consolidation=${(motivation.consolidation || 0).toFixed(3)}, ` +
           `curiosity=${(motivation.curiosity || 0).toFixed(3)}, ` +
           `claritySeeking=${(motivation.claritySeeking || 0).toFixed(3)}.`;
  }
}

