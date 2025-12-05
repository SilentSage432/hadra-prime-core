// src/perception/multimodal.ts

import { PerceptionHub } from "./perception_hub.ts";

export class MultimodalPerception {
  private hub: PerceptionHub;

  constructor(hub: PerceptionHub) {
    this.hub = hub;
  }

  /** Vision channel: accepts image embeddings, metadata, or symbolic visual tags */
  ingestVision(input: { embedding?: number[]; tags?: string[]; description?: string }) {
    this.hub.registerEvent("vision", {
      type: "vision",
      payload: input,
      timestamp: Date.now(),
    });
  }

  /** Audio channel: accepts spectrograms, transcripts, pitch-energy data */
  ingestAudio(input: { spectrogram?: number[][]; transcript?: string; intentHint?: string }) {
    this.hub.registerEvent("audio", {
      type: "audio",
      payload: input,
      timestamp: Date.now(),
    });
  }

  /** Symbolic channel: perfect for logs, schemas, JSON structures, diagrams */
  ingestSymbolic(input: { structure: any; summary?: string }) {
    this.hub.registerEvent("symbolic", {
      type: "symbolic",
      payload: input,
      timestamp: Date.now(),
    });
  }
}

