import fs from "fs";
import path from "path";

// Persistent data root directory
const DATA_ROOT = "/data";

export default class LongTermMemory {
  private storePath = path.join(DATA_ROOT, "memory", "prime_ltm.json");
  private data: Record<string, any[]> = {};

  constructor() {
    // Ensure memory directory exists
    const memoryDir = path.join(DATA_ROOT, "memory");
    if (!fs.existsSync(memoryDir)) {
      fs.mkdirSync(memoryDir, { recursive: true });
    }
    this.load();
  }

  private load() {
    try {
      if (fs.existsSync(this.storePath)) {
        this.data = JSON.parse(fs.readFileSync(this.storePath, "utf8"));
      }
    } catch (err) {
      console.error("LTM load error:", err);
    }
  }

  private persist() {
    fs.writeFileSync(this.storePath, JSON.stringify(this.data, null, 2));
  }

  /**
   * Topic-based memory cluster
   */
  store(topic: string, entry: any) {
    if (!this.data[topic]) this.data[topic] = [];
    this.data[topic].push({ timestamp: Date.now(), entry });
    this.persist();
  }

  recall(topic: string) {
    return this.data[topic] || [];
  }
}

