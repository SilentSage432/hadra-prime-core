import { VoiceModel } from "./voice.ts";
import type { ExpressionPacket } from "./types.ts";
import { SafetyGuard } from "../safety/safety_guard.ts";
import { MemoryStore } from "../memory/memory_store.ts";

export class ExpressionRouter {
  private voice: VoiceModel;
  private memory: MemoryStore;
  private safety: SafetyGuard;

  constructor(memory: MemoryStore, safety: SafetyGuard) {
    this.memory = memory;
    this.safety = safety;
    this.voice = new VoiceModel(safety, memory);
  }

  route(intent: any): ExpressionPacket {
    const { type, content } = intent;

    // System-level intents
    if (type === "status_check") {
      return this.voice.system("PRIME is online and stable.");
    }

    if (type === "error") {
      return this.voice.error(content || "Unknown error occurred.");
    }

    // Extract cognitive state if available
    const state = intent.cognitiveState;

    // Conversational or operator dialog
    const packet = this.voice.generate(type, content, state);

    // Add context continuity metadata
    if (intent.context?.latestSession) {
      packet.metadata = {
        ...packet.metadata,
        continuity: intent.context.latestSession,
      };
    }

    return packet;
  }
}

