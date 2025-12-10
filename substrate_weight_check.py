#!/usr/bin/env python3
"""
Substrate Weight Check - Parameter Magnitude + Init Sanity Check

Analyzes the parameter magnitudes in the MF-401→MF-500 substrate
to detect initialization issues, instabilities, or pathological values.
"""

import sys
import os

# Add src to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("❌ PyTorch is required for this check")
    print("   Install with: pip install torch")
    sys.exit(1)

try:
    from prime_core.influence_substrate import InfluenceSubstrateKernel
    SUBSTRATE_AVAILABLE = True
except ImportError as e:
    print(f"❌ Could not import InfluenceSubstrateKernel: {e}")
    sys.exit(1)

try:
    from neural.neural_bridge import NeuralBridge
    BRIDGE_AVAILABLE = True
except ImportError as e:
    print(f"❌ Could not import NeuralBridge: {e}")
    sys.exit(1)

dim = 128

print("=" * 60)
print("SUBSTRATE PARAMETER MAGNITUDE + INIT SANITY CHECK")
print("=" * 60)
print()

# Method 1: Try to get substrate through NeuralBridge
print("📦 Method 1: Loading substrate through NeuralBridge...")
try:
    nb = NeuralBridge()
    if hasattr(nb, 'mf_substrate') and nb.mf_substrate is not None:
        substrate = nb.mf_substrate
        print("✅ Substrate loaded successfully from NeuralBridge")
        detected_dim = nb.dim if hasattr(nb, 'dim') else dim
        print(f"   NeuralBridge dimension: {detected_dim}")
    else:
        print("⚠️ NeuralBridge.mf_substrate is None, creating direct instance...")
        substrate = InfluenceSubstrateKernel(dim=dim)
        print(f"✅ Created substrate directly with dim={dim}")
except Exception as e:
    print(f"⚠️ Could not load substrate through NeuralBridge: {e}")
    print("📦 Method 2: Creating substrate directly...")
    try:
        substrate = InfluenceSubstrateKernel(dim=dim)
        print(f"✅ Substrate created directly with dim={dim}")
    except Exception as e:
        print(f"❌ Could not create substrate: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\n" + "=" * 60)
print("ANALYZING PARAMETER MAGNITUDES...")
print("=" * 60)
print()

high_params = []
low_params = []
param_summary = []
total_params = 0
total_elements = 0

for name, param in substrate.named_parameters():
    if not param.requires_grad:
        continue
    
    total_params += 1
    num_elements = param.numel()
    total_elements += num_elements
    
    mean_val = float(param.data.mean())
    max_val = float(param.data.abs().max())
    min_val = float(param.data.abs().min())
    std_val = float(param.data.std())
    
    param_summary.append((name, mean_val, min_val, max_val, std_val, num_elements))
    
    # Identify extremes
    if max_val > 1.0:
        high_params.append((name, max_val))
    if min_val < 1e-12:
        low_params.append((name, min_val))

# Print summary
print(f"Total parameters: {total_params}")
print(f"Total parameter elements: {total_elements:,}")
print()

# Group by operator for cleaner output
operator_params = {}
for name, mean_val, min_val, max_val, std_val, num_elements in param_summary:
    # Extract operator name (e.g., "operators.0.linear1.weight" -> "MF401")
    parts = name.split('.')
    if len(parts) >= 2 and parts[0] == 'operators':
        op_idx = int(parts[1])
        op_name = f"MF{401 + op_idx}"
        if op_name not in operator_params:
            operator_params[op_name] = []
        operator_params[op_name].append((name, mean_val, min_val, max_val, std_val, num_elements))
    else:
        # Kernel-level parameters
        print(f"• {name}:")
        print(f"    mean={mean_val:.6f}, std={std_val:.6f}")
        print(f"    min={min_val:.6e}, max={max_val:.6e}")
        print(f"    elements={num_elements:,}")

# Print operator-level summary (sample first 5 operators)
print("\n" + "-" * 60)
print("OPERATOR PARAMETER SUMMARY (Sample: First 5 operators)")
print("-" * 60)
for op_name in sorted(operator_params.keys())[:5]:
    print(f"\n{op_name}:")
    for name, mean_val, min_val, max_val, std_val, num_elements in operator_params[op_name]:
        param_type = name.split('.')[-2] + '.' + name.split('.')[-1]
        print(f"  • {param_type}:")
        print(f"      mean={mean_val:.6f}, std={std_val:.6f}")
        print(f"      min={min_val:.6e}, max={max_val:.6e}")
        print(f"      elements={num_elements:,}")

if len(operator_params) > 5:
    print(f"\n... and {len(operator_params) - 5} more operators")

print("\n" + "=" * 60)
print("STABILITY REVIEW")
print("=" * 60)
print()

if len(high_params) == 0:
    print("✅ No high-magnitude instabilities detected.")
    print("   All parameters are within reasonable range (max < 1.0)")
else:
    print(f"⚠️ High-magnitude parameters found ({len(high_params)}):")
    for name, max_val in high_params[:10]:  # Show first 10
        print(f"   - {name}: max={max_val:.6f}")
    if len(high_params) > 10:
        print(f"   ... and {len(high_params) - 10} more")

if len(low_params) == 0:
    print("✅ No near-zero pathological values detected.")
    print("   All parameters have reasonable minimum magnitudes (min > 1e-12)")
else:
    print(f"⚠️ Extremely low-magnitude parameters found ({len(low_params)}):")
    for name, min_val in low_params[:10]:  # Show first 10
        print(f"   - {name}: min={min_val:.6e}")
    if len(low_params) > 10:
        print(f"   ... and {len(low_params) - 10} more")

# Additional checks
print("\n" + "-" * 60)
print("ADDITIONAL CHECKS")
print("-" * 60)

# Check for NaN or Inf
has_nan = False
has_inf = False
for name, param in substrate.named_parameters():
    if torch.isnan(param.data).any():
        print(f"❌ NaN detected in {name}")
        has_nan = True
    if torch.isinf(param.data).any():
        print(f"❌ Inf detected in {name}")
        has_inf = True

if not has_nan:
    print("✅ No NaN values detected")
if not has_inf:
    print("✅ No Inf values detected")

# Check parameter distribution
print("\n" + "-" * 60)
print("PARAMETER DISTRIBUTION SUMMARY")
print("-" * 60)
all_means = [m for _, m, _, _, _, _ in param_summary]
all_stds = [s for _, _, _, s, _, _ in param_summary]
all_maxes = [mx for _, _, _, mx, _, _ in param_summary]
all_mins = [mn for _, _, mn, _, _, _ in param_summary]

if all_means:
    print(f"Mean of parameter means: {sum(all_means) / len(all_means):.6f}")
    print(f"Mean of parameter stds: {sum(all_stds) / len(all_stds):.6f}")
    print(f"Overall max magnitude: {max(all_maxes):.6f}")
    print(f"Overall min magnitude: {min(all_mins):.6e}")

print("\n" + "=" * 60)
print("STABILITY CHECK COMPLETE")
print("=" * 60)

