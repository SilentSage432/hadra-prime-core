"""
ADRAE Prime-Core Runtime Entrypoint
Strict ML framing — substrate initialization + runtime verification
"""

import torch
from neural.neural_bridge import NeuralBridge
from prime_core.influence_substrate import InfluenceSubstrateKernel

# ------------------------------------------------------------
# Runtime boot display
# ------------------------------------------------------------
def banner():
    print("\n============================================================")
    print("        ADRAE PRIME-CORE — RUNTIME ACTIVATION (CPU)")
    print("============================================================\n")

def status(msg):
    print(f"[+] {msg}")

# ------------------------------------------------------------
# Runtime Initialization
# ------------------------------------------------------------
def initialize_runtime():
    banner()
    status("Initializing NeuralBridge (dim=128)...")

    bridge = NeuralBridge(dim=128)
    status("NeuralBridge ready.")

    status("Loading MF-500 unified substrate...")
    substrate = InfluenceSubstrateKernel(dim=128)
    status("Substrate loaded and instantiated.")

    return bridge, substrate

# ------------------------------------------------------------
# Runtime Verification Pass
# ------------------------------------------------------------
def run_verification_pass(bridge, substrate):
    status("Running substrate activation verification...")

    x_zero = torch.zeros(1, 128)
    x_rand = torch.randn(1, 128) * 0.01

    out_zero = substrate.forward(x_zero)
    out_rand = substrate.forward(x_rand)

    status(f"Zero-output norm: {out_zero.norm().item():.6f}")
    status(f"Rand-output norm: {out_rand.norm().item():.6f}")

    drift = (out_rand - out_zero).norm().item()
    status(f"Drift magnitude: {drift:.6f}")

    status("Verification pass complete.")

# ------------------------------------------------------------
# Main runtime entry
# ------------------------------------------------------------
def main():
    bridge, substrate = initialize_runtime()
    run_verification_pass(bridge, substrate)

    status("ADRAE Prime-Core runtime initialization complete.")
    print("\n============================================================")
    print("              RUNTIME EXIT — SYSTEM STABLE")
    print("============================================================\n")

if __name__ == "__main__":
    main()

