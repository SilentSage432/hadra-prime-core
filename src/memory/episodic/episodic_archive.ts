// src/memory/episodic/episodic_archive.ts
// A67: Episodic Memory - Episodic Archive

import type { Episode } from "./episode_builder.ts";

export class EpisodicArchive {
  private episodes: Episode[] = [];

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
}

