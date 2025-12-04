import { VoiceModel } from "./voice.ts";
import type { ExpressionPacket } from "./types.ts";
import { SafetyGuard } from "../safety/safety_guard.ts";
import { MemoryStore } from "../memory/memory_store.ts";
import { ExpressionModerator } from "./moderation/moderator.ts";
import { ToneDetector } from "./tone/tone_detector.ts";
import { StabilityMatrix } from "../stability/stability_matrix.ts";

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
      const systemMessage = this.voice.system("PRIME is online and stable.");
      return systemMessage;
    }

    if (type === "error") {
      const errorMessage = this.voice.error(content || "Unknown error occurred.");
      return errorMessage;
    }

    // Extract cognitive state if available
    const state = intent.cognitiveState;

    // Conversational or operator dialog
    const packet = this.voice.generate(type, content, state);

    // Determine tone profile from cognitive state or detect from content
    let toneProfile: "calm" | "technical" | "gentle" | "direct" | "neutral" = "neutral";
    if (state?.recommendedResponseMode === "direct") {
      toneProfile = "direct";
    } else if (state?.tone?.emotionalState === "calm") {
      toneProfile = "calm";
    } else if (state?.operatorFocus === "build" || state?.operatorFocus === "debug") {
      toneProfile = "technical";
    } else {
      const detectedTone = ToneDetector.detectTone(content || "");
      if (detectedTone.emotionalState === "focused") {
        toneProfile = "direct";
      } else if (detectedTone.emotionalState === "calm") {
        toneProfile = "calm";
      }
    }

    // Moderate the output message
    let moderatedMessage = ExpressionModerator.moderate(packet.message, toneProfile);

    // Add system awareness warnings when unstable
    if (StabilityMatrix.unstable()) {
      moderatedMessage = "[System Notice: PRIME operating in stability-protected mode]\n" + moderatedMessage;
    }

    // Update packet with moderated message
    packet.message = moderatedMessage;

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

