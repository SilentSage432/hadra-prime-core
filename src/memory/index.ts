import ShortTermMemory from "./short_term.ts";
import LongTermMemory from "./long_term.ts";

export default class MemoryBroker {
  public stm: ShortTermMemory;
  public ltm: LongTermMemory;

  constructor() {
    this.stm = new ShortTermMemory();
    this.ltm = new LongTermMemory();
  }

  /**
   * PRIME calls this for all internal events
   */
  remember(event: any) {
    this.stm.remember(event);
  }

  /**
   * Topic-based long-term storage
   */
  store(topic: string, entry: any) {
    this.ltm.store(topic, entry);
  }

  recallRecent() {
    return this.stm.getRecent();
  }

  recall(topic: string) {
    return this.ltm.recall(topic);
  }
}
