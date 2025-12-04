import type { IntentPacket } from "./types.ts";
import { ruleMatch } from "./rules.ts";
import { semanticScore } from "./semantic.ts";
import { fuseScores } from "./fusion.ts";

export class IntentEngine {
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
}

