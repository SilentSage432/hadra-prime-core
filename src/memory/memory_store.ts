/**
 * Memory Store - Wrapper for memory operations
 * Adapts MemoryLayer for use in expression system
 */
import { MemoryLayer } from "./memory.ts";

export class MemoryStore {
  private memoryLayer: MemoryLayer;

  constructor(memoryLayer: MemoryLayer) {
    this.memoryLayer = memoryLayer;
  }

  /**
   * Log an interaction to memory
   */
  logInteraction(type: string, data: any) {
    // Store as a general memory event
    // The MemoryLayer already handles interaction storage via storeInteraction
    // This is for additional logging
  }

  /**
   * Get recent interactions
   */
  getRecent(n: number = 5) {
    return this.memoryLayer.getRecent(n);
  }

  /**
   * Retrieve relevant memories based on intent type or topic
   */
  retrieveRelevant(topic: string) {
    const recent = this.memoryLayer.getRecent(10);
    // Filter by topic/intent type if available, otherwise return recent
    return recent.filter((record: any) => {
      if (record.intent?.type === topic) return true;
      if (record.summary?.toLowerCase().includes(topic.toLowerCase())) return true;
      return false;
    }).slice(0, 5); // Return top 5 relevant
  }
}

