// src/perception/perception_hub.ts

export class PerceptionHub {
  private buffer: any[] = [];
  private multimodalBuffer: any[] = [];

  registerEvent(channel: string, event: any) {
    if (["vision", "audio", "symbolic"].includes(channel)) {
      this.multimodalBuffer.push(event);
    }
    this.buffer.push(event);
  }

  getRecentEvents(limit: number = 10): any[] {
    return this.buffer.slice(-limit);
  }

  getMultimodalEvents(limit: number = 10): any[] {
    return this.multimodalBuffer.slice(-limit);
  }

  clear() {
    this.buffer = [];
    this.multimodalBuffer = [];
  }
}

