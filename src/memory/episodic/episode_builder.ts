// src/memory/episodic/episode_builder.ts
// A67: Episodic Memory - Episode Builder
// A115: Neural Event Segmentation Integration

import type { MicroEvent } from "./event_capture.ts";
import { NeuralEventSegmentation } from "../../cognition/neural_event_segmentation.ts";
import crypto from "crypto";

export interface Episode {
  id: string;
  title: string;
  startedAt: number;
  endedAt?: number;
  events: MicroEvent[];
  summary?: string;
  neuralEvents?: any[]; // A115: Neural event segments
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

    // A115: Attach neural events to episode
    const neuralEvents = NeuralEventSegmentation.getRecentEvents(3);
    if (neuralEvents.length > 0) {
      this.current.neuralEvents = neuralEvents;
      console.log("[PRIME-EPISODIC] Captured neural events for episode:", {
        episodeId: this.current.id,
        neuralEventCount: neuralEvents.length
      });
    }

    const finished = this.current;
    this.current = null;

    return finished;
  }

  getCurrent() {
    return this.current;
  }
}

