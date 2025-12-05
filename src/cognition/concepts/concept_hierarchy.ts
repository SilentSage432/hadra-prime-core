// src/cognition/concepts/concept_hierarchy.ts
// A76: Hierarchical Concept Networks

import type { Concept } from "./concept_engine.ts";
import { cosineSimilarity } from "../neural/similarity.ts";
import crypto from "crypto";

export interface MetaConcept {
  id: string;
  members: string[];
  strength: number;
  prototype: number[];
}

export interface Domain {
  id: string;
  concepts: string[];
  metaConcepts: string[];
  strength: number;
}

export class ConceptHierarchy {
  metaConcepts: MetaConcept[] = [];
  domains: Domain[] = [];

  /** Build meta-concepts by merging highly similar concepts */
  buildMetaConcepts(concepts: Concept[], threshold = 0.75) {
    if (concepts.length < 2) {
      this.metaConcepts = [];
      return;
    }

    const used = new Set<number>();
    const newMetaConcepts: MetaConcept[] = [];

    for (let i = 0; i < concepts.length; i++) {
      if (used.has(i)) continue;

      const group = [concepts[i]];
      used.add(i);

      for (let j = i + 1; j < concepts.length; j++) {
        if (used.has(j)) continue;

        const sim = cosineSimilarity(
          concepts[i].prototype,
          concepts[j].prototype
        );

        if (sim >= threshold) {
          group.push(concepts[j]);
          used.add(j);
        }
      }

      // Only create meta-concept if group has multiple members
      if (group.length > 1) {
        const proto = this.combinePrototypes(group);
        const metaId = crypto.randomUUID();

        newMetaConcepts.push({
          id: metaId,
          members: group.map((g) => g.id),
          strength: group.length,
          prototype: proto,
        });

        console.log(`[PRIME-META] Derived meta-concept ${metaId}: ${group.length} concepts`);
      }
    }

    this.metaConcepts = newMetaConcepts;
    console.log(`[PRIME-META] Total meta-concepts: ${this.metaConcepts.length}`);
  }

  /** Build domains by clustering meta-concepts */
  buildDomains(threshold = 0.70) {
    if (this.metaConcepts.length < 2) {
      this.domains = [];
      return;
    }

    const used = new Set<number>();
    const newDomains: Domain[] = [];

    for (let i = 0; i < this.metaConcepts.length; i++) {
      if (used.has(i)) continue;

      const group = [this.metaConcepts[i]];
      used.add(i);

      for (let j = i + 1; j < this.metaConcepts.length; j++) {
        if (used.has(j)) continue;

        const sim = cosineSimilarity(
          this.metaConcepts[i].prototype,
          this.metaConcepts[j].prototype
        );

        if (sim >= threshold) {
          group.push(this.metaConcepts[j]);
          used.add(j);
        }
      }

      // Only create domain if group has multiple meta-concepts
      if (group.length > 1) {
        const domainId = crypto.randomUUID();

        newDomains.push({
          id: domainId,
          concepts: group.flatMap((m) => m.members),
          metaConcepts: group.map((m) => m.id),
          strength: group.length,
        });

        console.log(`[PRIME-DOMAIN] Derived domain ${domainId}: ${group.length} meta-concepts`);
      }
    }

    this.domains = newDomains;
    console.log(`[PRIME-DOMAIN] Total domains: ${this.domains.length}`);
  }

  combinePrototypes(concepts: Concept[]): number[] {
    if (concepts.length === 0) return [];

    const len = concepts[0].prototype.length;
    const proto = new Array(len).fill(0);

    for (const c of concepts) {
      for (let i = 0; i < len; i++) {
        proto[i] += c.prototype[i];
      }
    }

    // Average the prototypes
    for (let i = 0; i < len; i++) {
      proto[i] /= concepts.length;
    }

    return proto;
  }

  /** Find domain that contains a given concept */
  findDomainForConcept(conceptId: string): Domain | null {
    return this.domains.find(
      (d) => d.concepts.includes(conceptId)
    ) || null;
  }

  /** Find domain that contains a given meta-concept */
  findDomainForMetaConcept(metaConceptId: string): Domain | null {
    return this.domains.find(
      (d) => d.metaConcepts.includes(metaConceptId)
    ) || null;
  }

  /** Get all domains */
  getDomains(): Domain[] {
    return [...this.domains];
  }

  /** Get all meta-concepts */
  getMetaConcepts(): MetaConcept[] {
    return [...this.metaConcepts];
  }
}

export const Hierarchy = new ConceptHierarchy();

