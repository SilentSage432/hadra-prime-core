// src/cognition/reflection_engine.ts
// A74: Neural Recall Integration
// A75: Concept Formation Integration
// A76: Hierarchical Concept Networks
// A77: Knowledge Graph Integration
// A78: Inference Engine Integration
// A79: Long-Range Predictive Model

import { Concepts } from "./concepts/concept_engine.ts";
import { Hierarchy } from "./concepts/concept_hierarchy.ts";
import { Knowledge } from "./knowledge/knowledge_graph.ts";
import { Inference } from "./inference/inference_engine.ts";
import { Foresight } from "./prediction/foresight_engine.ts";

export class ReflectionEngine {
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
    if (cognitiveState.embedding) {
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

