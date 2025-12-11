"""
ADRAE Prime-Core Runtime Monitor
Persistent stability logging and drift tracking
Strict ML framing — no cognitive semantics
"""

import os
import time
import json
import torch
from neural.neural_bridge import NeuralBridge
from prime_core.influence_substrate import InfluenceSubstrateKernel

LOG_DIR = "/data/logs"
STABILITY_DIR = "/data/stability"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(STABILITY_DIR, exist_ok=True)

health_state = {
    "initialized": False,
    "last_drift": None,
    "last_zero_norm": None,
    "last_rand_norm": None,
    "last_check": None,
}

bridge = None
substrate = None


def write_stability_snapshot(snapshot):
    """Write stability metrics snapshot to persistent storage."""
    ts = int(time.time())
    path = os.path.join(STABILITY_DIR, f"stability_{ts}.json")

    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)


def initialize_monitor(dim=128):
    """Initialize NeuralBridge and substrate for monitoring."""
    global bridge, substrate

    bridge = NeuralBridge(dim=dim)
    substrate = InfluenceSubstrateKernel(dim=dim)

    health_state["initialized"] = True
    return bridge, substrate


def run_health_check():
    """Run substrate health check and write stability snapshot."""
    if not health_state["initialized"]:
        initialize_monitor()

    x_zero = torch.zeros(1, 128)
    x_rand = torch.randn(1, 128) * 0.01

    out_zero = substrate.forward(x_zero)
    out_rand = substrate.forward(x_rand)

    zero_norm = out_zero.norm().item()
    rand_norm = out_rand.norm().item()
    drift_val = (out_rand - out_zero).norm().item()
    ts = time.time()

    snapshot = {
        "zero_norm": zero_norm,
        "rand_norm": rand_norm,
        "drift": drift_val,
        "timestamp": ts,
    }

    # Write snapshot to disk
    write_stability_snapshot(snapshot)

    health_state["last_zero_norm"] = zero_norm
    health_state["last_rand_norm"] = rand_norm
    health_state["last_drift"] = drift_val
    health_state["last_check"] = ts

    return {
        "status": "ok",
        **snapshot
    }

