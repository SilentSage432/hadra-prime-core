import type { IntentPacket } from "../intent_engine/types.ts";

export interface MemoryRecord {
  id: string;
  timestamp: Date;
  input: string;
  intent: IntentPacket;
  summary?: string;
  embedding?: number[];
}

export interface ShortTermMemory {
  buffer: MemoryRecord[];
  maxSize: number;
}

export interface LongTermMemory {
  records: MemoryRecord[];
}

