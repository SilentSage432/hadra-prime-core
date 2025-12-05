// src/kernel/threads/thread_monitor.ts

import { ThreadPool } from "./thread_pool.ts";

export class ThreadMonitor {
  static snapshot() {
    return ThreadPool.threads.map(t => ({
      id: t.id,
      active: t.active,
    }));
  }
}

