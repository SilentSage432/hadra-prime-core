import type { IntentPacket } from "./types.ts";
import { ruleMatch } from "./rules.ts";
import { semanticScore } from "./semantic.ts";
import { fuseScores } from "./fusion.ts";

export class IntentEngine {
  private currentIntent: string | null = null; // A107: Track current intent

  process(raw: string): IntentPacket {
    const rule = ruleMatch(raw);
    const semantic = semanticScore(raw);

    const fused = fuseScores(rule.type, rule.score, semantic.type, semantic.score);

    return {
      type: fused.finalType,
      confidence: fused.confidence,
      ruleScore: rule.score,
      semanticScore: semantic.score,
      raw,
      timestamp: new Date(),
      payload: {},
    };
  }

  // A107: Set top intent from harmonized resolution
  setTopIntent(intent: string) {
    this.currentIntent = intent;
  }
}

