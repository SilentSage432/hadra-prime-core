#!/usr/bin/env python3
"""
Static Drift-Spectrum Analysis (No Torch Required)

Analyzes the MF-phase operators and groups them into conceptual regions
to predict drift-stability profiles across the substrate pipeline.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent
substrate_file = ROOT / "src" / "prime_core" / "influence_substrate.py"

print("\n" + "=" * 60)
print("🔍 STATIC DRIFT–SPECTRUM ANALYSIS (NO TORCH REQUIRED)")
print("=" * 60)
print()

if not substrate_file.exists():
    print(f"❌ Substrate file not found: {substrate_file}")
    exit(1)

try:
    tree = ast.parse(substrate_file.read_text())
except Exception as e:
    print(f"❌ Error parsing substrate file: {e}")
    exit(1)

# Detect dynamically generated operators
ops = []
for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                name = target.id
                if name.startswith("MF") and name[2:].isdigit():
                    ops.append(name)

ops.sort(key=lambda x: int(x[2:]))  # Sort by number

print(f"Found {len(ops)} MF-phase operators.")
print(f"Expected: 100 (MF-401 → MF-500)")
print()

if len(ops) != 100:
    print(f"⚠️ Warning: Expected 100 operators, found {len(ops)}")
    print()

# Group by conceptual regions
regions = {
    "influence": [],
    "resonance": [],
    "coherence": [],
    "transport": [],
    "compression": [],
    "completion": []
}

for op in ops:
    n = int(op[2:])
    if 401 <= n <= 420:
        regions["influence"].append(op)
    elif 421 <= n <= 440:
        regions["resonance"].append(op)
    elif 441 <= n <= 460:
        regions["coherence"].append(op)
    elif 461 <= n <= 480:
        regions["transport"].append(op)
    elif 481 <= n <= 499:
        regions["compression"].append(op)
    elif n == 500:
        regions["completion"].append(op)
    else:
        print(f"⚠️ Operator {op} outside expected range")

# Print region completeness
print("-" * 60)
print("REGION COMPLETENESS")
print("-" * 60)
for region, arr in regions.items():
    expected_count = {
        "influence": 20,   # 401-420
        "resonance": 20,   # 421-440
        "coherence": 20,   # 441-460
        "transport": 20,   # 461-480
        "compression": 19, # 481-499
        "completion": 1    # 500
    }
    expected = expected_count.get(region, 0)
    status = "✅" if len(arr) == expected else "⚠️"
    print(f"{status} {region.upper():12} → {len(arr):2} operators (expected {expected})")

# Show sample operators from each region
print("\n" + "-" * 60)
print("REGION SAMPLES")
print("-" * 60)
for region, arr in regions.items():
    if arr:
        if len(arr) <= 5:
            sample = arr
        else:
            sample = [arr[0], arr[1], "...", arr[-2], arr[-1]]
        print(f"  {region.upper():12} → {', '.join(sample)}")

# Static stability heuristics:
def stability_score(name):
    """
    Calculate expected drift-stability score for an operator.
    
    Lower numbers = more sensitive to drift
    Mid-range = modulation/transport (moderate stability)
    High numbers = compression/consolidation (high stability)
    """
    n = int(name[2:])
    
    if n < 430:
        return 0.8   # influence fields moderate drift
    elif n < 460:
        return 0.7   # resonance/modulation tends to attenuate drift
    elif n < 480:
        return 0.6   # transport phases stabilize directional flow
    elif n < 499:
        return 0.9   # compression aggressively suppresses drift
    else:
        return 1.0   # completion phases finalize zero-drift state

print("\n" + "=" * 60)
print("📊 EXPECTED DRIFT-STABILITY PROFILE")
print("=" * 60)
print()

for region, arr in regions.items():
    if arr:
        scores = [stability_score(op) for op in arr]
        avg = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        print(f"  {region.upper():12} → stability factor: {avg:.3f} (range: {min_score:.3f} - {max_score:.3f})")

# Overall pipeline stability
print("\n" + "-" * 60)
print("OVERALL PIPELINE STABILITY")
print("-" * 60)
all_scores = [stability_score(op) for op in ops]
overall_avg = sum(all_scores) / len(all_scores)
overall_min = min(all_scores)
overall_max = max(all_scores)

print(f"  Average stability: {overall_avg:.3f}")
print(f"  Stability range:   {overall_min:.3f} - {overall_max:.3f}")

# Stability progression through pipeline
print("\n" + "-" * 60)
print("STABILITY PROGRESSION")
print("-" * 60)
print("  Phase Range    | Stability | Description")
print("  " + "-" * 55)
print("  401-420       |   0.800   | Influence propagation (moderate drift)")
print("  421-440       |   0.700   | Resonance modulation (drift attenuation)")
print("  441-460       |   0.700   | Coherence alignment (drift attenuation)")
print("  461-480       |   0.600   | Transport phases (directional flow)")
print("  481-499       |   0.900   | Compression (aggressive drift suppression)")
print("  500           |   1.000   | Completion (zero-drift finalization)")

print("\n" + "=" * 60)
print("STATIC SPECTRUM SCAN COMPLETE")
print("=" * 60)
print()

