// src/memory/episodic/episodic_archive.ts
// A67: Episodic Memory - Episodic Archive
// A115: Event Boundary support

import type { Episode } from "./episode_builder.ts";
import type { EventBoundary } from "./event_capture.ts";
import { EpisodeBuilder } from "./episode_builder.ts";

export class EpisodicArchive {
  private episodes: Episode[] = [];
  private boundaries: EventBoundary[] = [];
  private boundaryToEpisode: Map<string, string> = new Map(); // boundary ID -> episode ID

  store(episode: Episode) {
    this.episodes.push(episode);

    // prevent uncontrolled growth
    if (this.episodes.length > 1000) {
      this.episodes.shift();
    }
  }

  list() {
    return [...this.episodes];
  }

  getLatest() {
    return this.episodes[this.episodes.length - 1] || null;
  }

  findByTitle(title: string) {
    return this.episodes.filter(ep => ep.title === title);
  }

  /**
   * A115: Capture event boundary and link to active episode
   */
  captureBoundary(boundary: EventBoundary): void {
    this.boundaries.push(boundary);

    // Maintain bounded history
    if (this.boundaries.length > 1000) {
      this.boundaries.shift();
    }

    // Link boundary to active episode
    this.linkBoundaryToActiveEpisode(boundary);

    console.log("[PRIME-EPISODIC] Event boundary captured:", {
      id: boundary.id,
      reason: boundary.reason,
      linkedToEpisode: this.boundaryToEpisode.get(boundary.id) || "none"
    });
  }

  /**
   * A115: Link boundary to active episode
   */
  private linkBoundaryToActiveEpisode(boundary: EventBoundary): void {
    // Get episodeBuilder from global scope (set in kernel)
    const episodeBuilder = (globalThis as any).__PRIME_EPISODE_BUILDER__;
    if (!episodeBuilder) {
      // Fallback: try to get from kernel instance
      const kernelInstance = (globalThis as any).__PRIME_KERNEL__;
      if (kernelInstance && kernelInstance.getCurrentEpisode) {
        const currentEpisode = kernelInstance.getCurrentEpisode();
        if (currentEpisode) {
          this.boundaryToEpisode.set(boundary.id, currentEpisode.id);
        }
      }
      return;
    }

    const currentEpisode = episodeBuilder.getCurrent();
    if (currentEpisode) {
      this.boundaryToEpisode.set(boundary.id, currentEpisode.id);
      
      // Add boundary metadata to episode if it has a boundaries field
      if ((currentEpisode as any).boundaries) {
        (currentEpisode as any).boundaries.push(boundary.id);
      }
    }
  }

  /**
   * A115: Get boundaries for an episode
   */
  getBoundariesForEpisode(episodeId: string): EventBoundary[] {
    const boundaryIds: string[] = [];
    this.boundaryToEpisode.forEach((epId, boundaryId) => {
      if (epId === episodeId) {
        boundaryIds.push(boundaryId);
      }
    });

    return this.boundaries.filter(b => boundaryIds.includes(b.id));
  }

  /**
   * A115: Get recent boundaries
   */
  getRecentBoundaries(count: number = 10): EventBoundary[] {
    return this.boundaries.slice(-count);
  }

  /**
   * A115: Get all boundaries
   */
  getAllBoundaries(): EventBoundary[] {
    return [...this.boundaries];
  }
}

