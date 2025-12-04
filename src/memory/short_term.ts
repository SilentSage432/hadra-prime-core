export default class ShortTermMemory {
  private buffer: Array<{ timestamp: number; data: any }> = [];
  private limit: number = 40; // adaptive later

  constructor() {}

  remember(data: any) {
    this.buffer.push({ timestamp: Date.now(), data });
    if (this.buffer.length > this.limit) this.buffer.shift();
  }

  getRecent() {
    return [...this.buffer];
  }

  clear() {
    this.buffer = [];
  }
}

