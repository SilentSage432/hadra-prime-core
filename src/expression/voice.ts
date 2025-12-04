import type { ExpressionPacket } from "./types.ts";
import { SafetyGuard } from "../safety/safety_guard.ts";
import { MemoryStore } from "../memory/memory_store.ts";
import { ToneEngine } from "./tone/tone_engine.ts";

export class VoiceModel {
  private tone: ToneEngine;
  private safety: SafetyGuard;
  private memory: MemoryStore;

  constructor(safety: SafetyGuard, memory: MemoryStore) {
    this.safety = safety;
    this.memory = memory;
    this.tone = new ToneEngine();
  }

  generate(intentType: string, content: string, state?: any): ExpressionPacket {
    const sanitized = this.safety.filterOutput(content);

    // Use cognitive state if available, otherwise fall back to tone analysis
    const toneVector = state?.tone || this.tone.analyze(content);

    // Tone shaping
    let shaped = this.tone.shapeOutput(sanitized, toneVector);

    // Meta-aware messaging adjustments
    if (state?.meta) {
      const meta = state.meta;
      if (meta.actionRecommendation === "clarify") {
        shaped = `I can answer, but I need a bit more information. ${meta.notes.join(" ")} ${shaped}`;
      } else if (meta.actionRecommendation === "warn") {
        shaped = `I'll answer, but with caution — ${meta.notes.join(" ")} ${shaped}`;
      }
    }

    // Log this output
    this.memory.logInteraction("expression", {
      intent: intentType,
      output: shaped,
      cognitiveState: state,
    });

    // Use cognitive state metadata if available
    if (state) {
      return {
        type: "reply",
        message: shaped,
        confidence: state.meta?.certaintyLevel ?? state.priorityLevel,
        metadata: {
          tone: state.tone,
          operatorFocus: state.operatorFocus,
          responseMode: state.recommendedResponseMode,
          context: state.context,
          meta: state.meta,
        },
      };
    }

    // Fallback for when cognitive state is not available
    return {
      type: "reply",
      message: shaped,
      confidence: 0.95,
      metadata: {
        tone: toneVector,
        model: "prime-adaptive-voice-v1",
      },
    };
  }

  system(message: string): ExpressionPacket {
    return {
      type: "system",
      message: this.safety.filterOutput(message),
      confidence: 1.0,
    };
  }

  error(message: string): ExpressionPacket {
    return {
      type: "error",
      message,
      confidence: 0.7,
    };
  }
}

