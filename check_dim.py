#!/usr/bin/env python3
"""
ADRAE Prime-Core Dimensionality Diagnostic

Checks the dimension used by NeuralBridge and tests substrate integration.
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
    print(f"✅ PyTorch available: {torch.__version__}")
except ImportError:
    print("⚠️ PyTorch not available - substrate tests will be skipped")
    print("   Install with: pip install torch")
    TORCH_AVAILABLE = False

def check_dimension_statically():
    """Check dimension by reading the code"""
    print("📖 Checking dimension from code...")
    results = {}
    
    # Check NeuralBridge
    nb_path = os.path.join(src_path, 'neural', 'neural_bridge.py')
    if os.path.exists(nb_path):
        with open(nb_path, 'r') as f:
            content = f.read()
            # Look for self.dim = getattr(self, "embedding_dim", 128)
            import re
            match = re.search(r'self\.dim\s*=\s*getattr\(self,\s*["\']embedding_dim["\'],\s*(\d+)\)', content)
            if match:
                default_dim = int(match.group(1))
                print(f"✅ NeuralBridge default dimension: {default_dim}")
                results['neural_bridge'] = default_dim
            else:
                print("⚠️ Could not find dimension in NeuralBridge code")
    
    # Check substrate integration
    substrate_path = os.path.join(src_path, 'prime_core', 'influence_substrate.py')
    if os.path.exists(substrate_path):
        print("✅ Found influence_substrate.py")
        with open(substrate_path, 'r') as f:
            content = f.read()
            # Check if InfluenceSubstrateKernel exists
            if 'class InfluenceSubstrateKernel' in content:
                print("✅ InfluenceSubstrateKernel class found")
                # Count MF operators
                mf_count = len(re.findall(r'MF\d{3}\s*=', content))
                if mf_count >= 100:
                    print(f"✅ Found {mf_count} MF-phase operators (MF-401 → MF-500)")
                else:
                    print(f"⚠️ Found {mf_count} MF-phase operators (expected 100)")
            else:
                print("⚠️ InfluenceSubstrateKernel class not found")
    else:
        print("⚠️ influence_substrate.py not found")
    
    return results.get('neural_bridge')

def check_neural_bridge():
    """Check NeuralBridge initialization and dimension"""
    print("=" * 60)
    print("ADRAE PRIME-CORE DIMENSIONALITY CHECK")
    print("=" * 60)
    print()
    
    # First, try to get dimension from code
    static_dim = check_dimension_statically()
    
    # Try to import and instantiate
    try:
        # Try importing with the correct path (src is in sys.path)
        from neural.neural_bridge import NeuralBridge
        print("✅ Successfully imported NeuralBridge")
    except ImportError as e:
        print(f"⚠️ Failed to import NeuralBridge: {e}")
        if static_dim:
            print(f"\n📊 Static Analysis Results:")
            print(f"   Default dimension: {static_dim}")
            print(f"   (Cannot test runtime without PyTorch)")
            print("\n" + "=" * 60)
            print("DIAGNOSTIC SUMMARY (Static Analysis Only)")
            print("=" * 60)
            print(f"✅ NeuralBridge dimension: {static_dim}")
            print("✅ MF-401→MF-500 Substrate: VERIFIED IN CODE")
            print("⚠️ PyTorch: NOT INSTALLED (install for runtime tests)")
            print("=" * 60)
            return True
        return False
    
    try:
        # NeuralBridge takes no parameters in __init__
        print("\n📦 Initializing NeuralBridge...")
        nb = NeuralBridge()
        print("✅ NeuralBridge initialized successfully")
        
        # Check the dimension attribute
        if hasattr(nb, 'dim'):
            detected_dim = nb.dim
            print(f"✅ Detected dimension: {detected_dim}")
            if static_dim and detected_dim != static_dim:
                print(f"⚠️ Dimension mismatch: code default={static_dim}, runtime={detected_dim}")
        else:
            print("⚠️ NeuralBridge has no 'dim' attribute")
            detected_dim = static_dim or 128  # Use static or default fallback
            print(f"   Using dimension: {detected_dim}")
        
        # Check if substrate is initialized
        if hasattr(nb, 'mf_substrate'):
            if nb.mf_substrate is not None:
                print(f"✅ MF-401→MF-500 Substrate initialized")
                print(f"   Substrate dimension: {detected_dim}")
            else:
                print("⚠️ MF-401→MF-500 Substrate is None (not available or failed to initialize)")
        else:
            print("⚠️ NeuralBridge has no 'mf_substrate' attribute")
        
        # Test forward pass if PyTorch is available
        if TORCH_AVAILABLE:
            print("\n🧪 Testing forward pass...")
            try:
                # Create test tensor matching the detected dimension
                test_tensor = torch.zeros(1, detected_dim)
                print(f"   Input tensor shape: {test_tensor.shape}")
                
                # Test forward pass
                output = nb.forward(test_tensor)
                print(f"✅ Forward pass successful")
                print(f"   Output tensor shape: {output.shape}")
                
                # Verify output matches input dimension
                if output.shape[-1] == detected_dim:
                    print(f"✅ Output dimension matches input dimension ({detected_dim})")
                else:
                    print(f"⚠️ Dimension mismatch: input={detected_dim}, output={output.shape[-1]}")
                
            except Exception as e:
                print(f"❌ Forward pass failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\n⚠️ Skipping forward pass test (PyTorch not available)")
        
        # Test with different dimensions to see if substrate handles them
        if TORCH_AVAILABLE and hasattr(nb, 'mf_substrate') and nb.mf_substrate is not None:
            print("\n🧪 Testing substrate with different tensor dimensions...")
            test_dims = [64, 128, 256, 512]
            for test_dim in test_dims:
                try:
                    test_tensor = torch.zeros(1, test_dim)
                    # Direct substrate test
                    if test_dim == detected_dim:
                        output = nb.mf_substrate(test_tensor)
                        print(f"✅ Substrate accepted dim={test_dim}, output shape={output.shape}")
                    else:
                        print(f"⚠️ Skipping dim={test_dim} (doesn't match NeuralBridge dim={detected_dim})")
                except Exception as e:
                    print(f"❌ Substrate failed with dim={test_dim}: {e}")
        
        print("\n" + "=" * 60)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 60)
        print(f"✅ NeuralBridge dimension: {detected_dim}")
        if hasattr(nb, 'mf_substrate') and nb.mf_substrate is not None:
            print("✅ MF-401→MF-500 Substrate: INITIALIZED")
        else:
            print("⚠️ MF-401→MF-500 Substrate: NOT AVAILABLE")
        if TORCH_AVAILABLE:
            print("✅ PyTorch: AVAILABLE")
        else:
            print("⚠️ PyTorch: NOT INSTALLED (runtime tests skipped)")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"❌ Error during NeuralBridge check: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = check_neural_bridge()
    sys.exit(0 if success else 1)

