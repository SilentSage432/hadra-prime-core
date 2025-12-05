// src/memory/episodic/episode_builder.ts
// A67: Episodic Memory - Episode Builder

import type { MicroEvent } from "./event_capture.ts";
import crypto from "crypto";

export interface Episode {
  id: string;
  title: string;
  startedAt: number;
  endedAt?: number;
  events: MicroEvent[];
  summary?: string;
}

export class EpisodeBuilder {
  private current: Episode | null = null;

  startEpisode(title: string) {
    this.current = {
      id: crypto.randomUUID(),
      title,
      startedAt: Date.now(),
      events: []
    };

    return this.current;
  }

  addEvent(event: MicroEvent) {
    if (!this.current) return;

    this.current.events.push(event);
  }

  closeEpisode(summary?: string) {
    if (!this.current) return null;

    this.current.endedAt = Date.now();
    if (summary) this.current.summary = summary;

    const finished = this.current;
    this.current = null;

    return finished;
  }

  getCurrent() {
    return this.current;
  }
}

