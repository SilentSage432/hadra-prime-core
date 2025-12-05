// src/cognition/reflection_engine.ts
// A74: Neural Recall Integration
// A75: Concept Formation Integration
// A76: Hierarchical Concept Networks
// A77: Knowledge Graph Integration

import { Concepts } from "./concepts/concept_engine.ts";
import { Hierarchy } from "./concepts/concept_hierarchy.ts";
import { Knowledge } from "./knowledge/knowledge_graph.ts";

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

