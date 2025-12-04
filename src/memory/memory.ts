import type { MemoryRecord, ShortTermMemory, LongTermMemory } from "./types.ts";
import type { IntentPacket } from "../intent_engine/types.ts";
import crypto from "crypto";

export class MemoryLayer {
  private shortTerm: ShortTermMemory = {
    buffer: [],
    maxSize: 20, // last 20 turns remembered
  };

  private longTerm: LongTermMemory = {
    records: [],
  };

  storeInteraction(input: string, intent: IntentPacket) {
    const record: MemoryRecord = {
      id: crypto.randomUUID(),
      timestamp: new Date(),
      input,
      intent,
      summary: this.generateSummary(intent),
      embedding: this.placeholderEmbedding(intent),
    };

    this.shortTerm.buffer.push(record);

    if (this.shortTerm.buffer.length > this.shortTerm.maxSize) {
      const removed = this.shortTerm.buffer.shift();
      if (removed) this.longTerm.records.push(removed);
    }
  }

  getRecent(n: number = 5): MemoryRecord[] {
    return this.shortTerm.buffer.slice(-n);
  }

  /**
   * Placeholder summary generator.
   * Real version will use PRIME's ML summarizer.
   */
  generateSummary(intent: IntentPacket): string {
    return `${intent.type} intent @ ${intent.timestamp.toISOString()}`;
  }

  /**
   * Placeholder embedding function.
   * Real version will call Rust/WASM or Python embeddings.
   */
  placeholderEmbedding(intent: IntentPacket): number[] {
    return [intent.confidence, intent.ruleScore, intent.semanticScore];
  }
}

