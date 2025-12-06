"""
Task Queue Manager (A154)
--------------------------
Manages PRIME's task queue with priority-based ordering.
"""

import heapq
import time


class TaskQueue:

    def __init__(self):
        # priority queue: (priority, timestamp, task)
        self.queue = []
        self.counter = 0  # ensures ordering

    def add_task(self, text, priority=5):
        """
        Priority scale: 1 = high, 10 = low
        """
        timestamp = time.time()
        entry = (priority, timestamp, {"text": text, "priority": priority})
        heapq.heappush(self.queue, entry)
        return entry

    def get_next_task(self):
        if not self.queue:
            return None
        priority, ts, task = heapq.heappop(self.queue)
        return task

    def peek(self):
        if not self.queue:
            return None
        return self.queue[0][2]

