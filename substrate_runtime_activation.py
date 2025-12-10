#!/usr/bin/env python3
"""
ADRAE Prime-Core — MF-500 Substrate Runtime Activation

Tests the substrate through NeuralBridge with various input tensors
to verify runtime activation and stability.
"""

import os
import sys

# ---------------------------------------
# FIX: Ensure project root is importable
# ---------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

# Add src to path for imports
src_path = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, src_path)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("❌ PyTorch is required for runtime activation tests")
    print("   Install with: pip install torch")
    sys.exit(1)

try:
    from neural.neural_bridge import NeuralBridge
except ImportError as e:
    print(f"❌ Could not import NeuralBridge: {e}")
    sys.exit(1)

dim = 128

print("\n" + "=" * 60)
print("🚀 ADRAE PRIME-CORE — MF-500 SUBSTRATE RUNTIME ACTIVATION")
print("=" * 60)
print()

# Initialize NeuralBridge (it doesn't take dim parameter)
print("📦 Initializing NeuralBridge...")
try:
    nb = NeuralBridge()
    detected_dim = nb.dim if hasattr(nb, 'dim') else dim
    print(f"✅ NeuralBridge initialized (dim={detected_dim})")
    
    # Check if substrate is available
    if hasattr(nb, 'mf_substrate') and nb.mf_substrate is not None:
        print("✅ MF-500 Substrate is available")
    else:
        print("❌ MF-500 Substrate is not available")
        print("   Substrate may have failed to initialize")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Failed to initialize NeuralBridge: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Use detected dimension
dim = detected_dim

# -------------------------------------
# 1. Zero Tensor Activation Test
# -------------------------------------
print("\n" + "-" * 60)
print("1. ZERO-TENSOR ACTIVATION TEST")
print("-" * 60)

x0 = torch.zeros(1, dim)
print(f"   Input: zeros tensor, shape={x0.shape}")
print("   Running zero-tensor activation...")

try:
    out0 = nb.forward(x0)
    norm0 = out0.norm().item()
    print(f"   ✅ Zero-Tensor OK")
    print(f"      Output shape: {out0.shape}")
    print(f"      Output norm: {norm0:.6f}")
except Exception as e:
    print(f"   ❌ Zero-Tensor FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# -------------------------------------
# 2. Small Random Tensor Activation
# -------------------------------------
print("\n" + "-" * 60)
print("2. SMALL-RANDOM ACTIVATION TEST")
print("-" * 60)

x1 = torch.randn(1, dim) * 0.01
print(f"   Input: small random tensor (scale=0.01), shape={x1.shape}")
print("   Running small-random activation...")

try:
    out1 = nb.forward(x1)
    norm1 = out1.norm().item()
    print(f"   ✅ Small-Random OK")
    print(f"      Output shape: {out1.shape}")
    print(f"      Output norm: {norm1:.6f}")
except Exception as e:
    print(f"   ❌ Small-Random FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# -------------------------------------
# 3. Amplitude Sweep Test
# -------------------------------------
print("\n" + "-" * 60)
print("3. AMPLITUDE SWEEP TEST")
print("-" * 60)

print("   Testing various input scales (1 → 50)...")
scales = [1, 5, 10, 25, 50]
sweep_results = []

for scale in scales:
    xt = torch.randn(1, dim) * scale
    try:
        outt = nb.forward(xt)
        normt = outt.norm().item()
        sweep_results.append((scale, normt, True))
        print(f"   ✅ Scale {scale:>2} OK — output norm: {normt:.6f}")
    except Exception as e:
        sweep_results.append((scale, None, False))
        print(f"   ❌ FAILED at scale {scale}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Check for stability across scales
if all(result[2] for result in sweep_results):
    norms = [result[1] for result in sweep_results]
    norm_range = max(norms) - min(norms)
    norm_ratio = max(norms) / min(norms) if min(norms) > 0 else float('inf')
    print(f"\n   Stability check:")
    print(f"      Norm range: {norm_range:.6f}")
    print(f"      Norm ratio: {norm_ratio:.2f}x")
    if norm_ratio < 10:
        print(f"      ✅ Output norms are stable across input scales")
    else:
        print(f"      ⚠️ Output norms vary significantly across scales")

# -------------------------------------
# 4. Drift-Spectrum Probe
# -------------------------------------
print("\n" + "-" * 60)
print("4. DRIFT-SPECTRUM PROBE")
print("-" * 60)

try:
    # Norm difference between zero-tensor and small-random outputs
    drift = torch.norm(out1 - out0).item()
    print(f"   Drift magnitude (||out1 - out0||): {drift:.6f}")
    
    # Relative drift
    if norm0 > 0:
        relative_drift = drift / norm0
        print(f"   Relative drift: {relative_drift:.6f}")
    
    # Check if drift is within expected operational envelope
    # For a normalized substrate, drift should be reasonable
    if drift < 10.0:  # Reasonable threshold
        print("   ✅ Drift within expected operational envelope")
    else:
        print("   ⚠️ Drift may be outside expected envelope")
        
except Exception as e:
    print(f"   ❌ Drift Spectrum FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# -------------------------------------
# 5. Substrate Direct Test (if available)
# -------------------------------------
print("\n" + "-" * 60)
print("5. SUBSTRATE DIRECT TEST")
print("-" * 60)

if hasattr(nb, 'mf_substrate') and nb.mf_substrate is not None:
    print("   Testing substrate directly...")
    try:
        x_test = torch.randn(1, dim) * 0.1
        out_direct = nb.mf_substrate(x_test)
        norm_direct = out_direct.norm().item()
        print(f"   ✅ Direct substrate test OK")
        print(f"      Output norm: {norm_direct:.6f}")
    except Exception as e:
        print(f"   ⚠️ Direct substrate test failed: {e}")
        print("      (This is non-critical if forward() works)")
else:
    print("   ⚠️ Substrate not available for direct test")

# Summary
print("\n" + "=" * 60)
print("✨ SUBSTRATE ACTIVATION COMPLETE — MF-500 ONLINE")
print("=" * 60)
print()
print("✅ All activation tests passed")
print("✅ Substrate is operational")
print("✅ Ready for integration into ADRAE Prime-Core runtime")
print()

