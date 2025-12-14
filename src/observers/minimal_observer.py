# src/observers/minimal_observer.py

import time
import random

BASE_INTERVAL = 5.0     # seconds (≥ 10× cognitive tick)
JITTER = 0.5            # seconds


def observer_loop(read_state_fn):
    """
    Ephemeral observer loop.
    Prints anonymized scalar motion only.
    """

    while True:
        sleep_time = BASE_INTERVAL + random.uniform(-JITTER, JITTER)
        time.sleep(max(0.1, sleep_time))

        try:
            state = read_state_fn()

            snapshot = {
                "ts": time.time(),
                "drift": state["drift"],
                "coherence": state["coherence"],
                "attention_entropy": state["attention_entropy"]
            }

            print(snapshot)

        except Exception as e:
            # Observer must never affect runtime
            print({"observer_error": str(e)})

