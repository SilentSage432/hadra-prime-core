#!/usr/bin/env python3
"""
Static Substrate Routing Validation (No Torch Needed)

Validates the substrate structure and integration without requiring PyTorch.
Uses AST parsing to check code structure statically.
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Locate substrate file
substrate_file = ROOT / "src" / "prime_core" / "influence_substrate.py"
bridge_file = ROOT / "src" / "neural" / "neural_bridge.py"

print("\n" + "=" * 60)
print("🔍 STATIC SUBSTRATE ROUTING VALIDATION")
print("=" * 60)
print()

# Check if files exist
if not substrate_file.exists():
    print(f"❌ Substrate file not found: {substrate_file}")
    sys.exit(1)

if not bridge_file.exists():
    print(f"❌ Bridge file not found: {bridge_file}")
    sys.exit(1)

print(f"✅ Found substrate file: {substrate_file.name}")
print(f"✅ Found bridge file: {bridge_file.name}")
print()

# --- Helper: extract class methods from AST ---
def get_class_methods(tree, class_name):
    """Extract method names from a class definition"""
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            methods = []
            for n in node.body:
                if isinstance(n, ast.FunctionDef):
                    methods.append(n.name)
            return methods
    return []

# --- Helper: extract class names from AST ---
def get_class_names(tree):
    """Extract all class names from AST"""
    classes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)
    return classes

# --- Helper: extract MF operator assignments ---
def get_mf_operator_assignments(tree):
    """Extract MF operator variable assignments (e.g., MF401 = build_mf_operator_class(401))"""
    operators = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith("MF") and target.id[2:].isdigit():
                    operators.append(target.id)
    return operators

# --- Helper: check for attribute usage ---
def check_attribute_usage(tree, attr_name):
    """Check if an attribute is used in the code"""
    class AttributeVisitor(ast.NodeVisitor):
        def __init__(self):
            self.found = False
        
        def visit_Attribute(self, node):
            if isinstance(node, ast.Attribute) and node.attr == attr_name:
                self.found = True
            self.generic_visit(node)
    
    visitor = AttributeVisitor()
    visitor.visit(tree)
    return visitor.found

# --- Parse substrate operators ---
print("-" * 60)
print("1. VALIDATING MF-PHASE OPERATORS")
print("-" * 60)

try:
    substrate_source = substrate_file.read_text()
    tree = ast.parse(substrate_source)
    
    # Get MF operators from variable assignments (they're created dynamically)
    operators = get_mf_operator_assignments(tree)
    operators.sort(key=lambda x: int(x[2:]))  # Sort by number
    
    print(f"Found {len(operators)} MF-phase operators.")
    print(f"Expected: 100 (MF-401 → MF-500)")
    print()
    
    if len(operators) == 0:
        print("❌ No MF operators found!")
    else:
        print(f"First 5 operators: {operators[:5]}")
        print(f"Last 5 operators: {operators[-5:]}")
        print()
        
        # Check for missing operators
        missing = []
        for i in range(401, 501):
            name = f"MF{i}"
            if name not in operators:
                missing.append(name)
        
        if missing:
            print(f"❌ Missing {len(missing)} operators:")
            if len(missing) <= 10:
                print(f"   {missing}")
            else:
                print(f"   {missing[:10]} ... and {len(missing) - 10} more")
        else:
            print("✅ All 100 operators present (MF-401 → MF-500)")
        
        # Check for extra operators
        extra = [op for op in operators if not (401 <= int(op[2:]) <= 500)]
        if extra:
            print(f"⚠️ Found {len(extra)} operators outside expected range:")
            print(f"   {extra}")

except Exception as e:
    print(f"❌ Error parsing substrate file: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# --- Validate InfluenceSubstrateKernel structure ---
print("\n" + "-" * 60)
print("2. VALIDATING INFLUENCESUBSTRATEKERNEL STRUCTURE")
print("-" * 60)

kernel_methods = get_class_methods(tree, "InfluenceSubstrateKernel")
expected_methods = ["__init__", "forward"]

print(f"Kernel methods found: {kernel_methods}")
print(f"Expected methods: {expected_methods}")
print()

if all(m in kernel_methods for m in expected_methods):
    print("✅ Kernel structure OK - all required methods present")
else:
    missing_methods = [m for m in expected_methods if m not in kernel_methods]
    print(f"❌ Kernel missing required methods: {missing_methods}")

# Check for MFPhaseBase
base_methods = get_class_methods(tree, "MFPhaseBase")
if base_methods:
    print(f"✅ MFPhaseBase found with methods: {base_methods}")
else:
    print("⚠️ MFPhaseBase not found or has no methods")

# Check for normalize_tensor function
has_normalize = any(
    isinstance(node, ast.FunctionDef) and node.name == "normalize_tensor"
    for node in tree.body
)
if has_normalize:
    print("✅ normalize_tensor function found")
else:
    print("⚠️ normalize_tensor function not found")

# --- Validate neural_bridge.py integration ---
print("\n" + "-" * 60)
print("3. VALIDATING NEURALBRIDGE INTEGRATION")
print("-" * 60)

try:
    bridge_source = bridge_file.read_text()
    bridge_tree = ast.parse(bridge_source)
    
    bridge_methods = get_class_methods(bridge_tree, "NeuralBridge")
    
    print(f"NeuralBridge methods discovered: {len(bridge_methods)} methods")
    if "forward" in bridge_methods:
        print("✅ NeuralBridge has forward() method")
    else:
        print("⚠️ NeuralBridge forward() method not found")
    
    # Check for substrate import
    has_substrate_import = "InfluenceSubstrateKernel" in bridge_source or "influence_substrate" in bridge_source
    if has_substrate_import:
        print("✅ Substrate import found in NeuralBridge")
    else:
        print("⚠️ Substrate import not found in NeuralBridge")
    
    # Check for mf_substrate attribute
    has_mf_substrate = "mf_substrate" in bridge_source
    if has_mf_substrate:
        print("✅ mf_substrate attribute found in NeuralBridge")
        
        # Check if it's used in forward
        if "forward" in bridge_source:
            # Find the forward method more accurately
            forward_idx = bridge_source.find("def forward(self, x):")
            if forward_idx == -1:
                forward_idx = bridge_source.find("def forward(self")
            
            if forward_idx != -1:
                # Look for the next method definition or end of class
                next_method = bridge_source.find("\n    def ", forward_idx + 1)
                if next_method == -1:
                    # Look for end of class or next top-level def
                    next_method = bridge_source.find("\ndef ", forward_idx + 1)
                
                if next_method != -1:
                    forward_section = bridge_source[forward_idx:next_method]
                else:
                    # Take a larger section if we can't find the end
                    forward_section = bridge_source[forward_idx:forward_idx+1000]
                
                if "mf_substrate" in forward_section:
                    print("✅ mf_substrate is used in forward() method")
                else:
                    print("⚠️ mf_substrate not found in forward() method")
                    # Double-check with a simpler search
                    if "self.mf_substrate" in bridge_source:
                        print("   (but self.mf_substrate exists elsewhere in the file)")
            else:
                print("⚠️ Could not locate forward() method definition")
    else:
        print("❌ mf_substrate attribute not found in NeuralBridge")
    
    # Check for substrate initialization
    has_substrate_init = "InfluenceSubstrateKernel" in bridge_source and "mf_substrate" in bridge_source
    if has_substrate_init:
        print("✅ Substrate initialization code found")
    else:
        print("⚠️ Substrate initialization code not clearly found")

except Exception as e:
    print(f"❌ Error parsing bridge file: {e}")
    import traceback
    traceback.print_exc()

# --- Check for build_mf_operator_class function ---
print("\n" + "-" * 60)
print("4. VALIDATING OPERATOR GENERATION")
print("-" * 60)

has_build_function = any(
    isinstance(node, ast.FunctionDef) and node.name == "build_mf_operator_class"
    for node in tree.body
)
if has_build_function:
    print("✅ build_mf_operator_class function found")
else:
    print("⚠️ build_mf_operator_class function not found")

# Check if operators are created using the build function
has_operator_creation = "build_mf_operator_class" in substrate_source
if has_operator_creation:
    print("✅ Operator creation using build_mf_operator_class detected")
else:
    print("⚠️ Operator creation method unclear")

# --- Summary ---
print("\n" + "=" * 60)
print("STATIC ROUTING VALIDATION COMPLETE")
print("=" * 60)
print()

# Final status
all_operators_present = len(missing) == 0 if 'missing' in locals() else False
kernel_ok = all(m in kernel_methods for m in expected_methods) if 'kernel_methods' in locals() else False
bridge_integration = has_mf_substrate if 'has_mf_substrate' in locals() else False

if all_operators_present and kernel_ok and bridge_integration:
    print("✅ Overall Status: VALIDATION PASSED")
    print("   - All 100 MF operators present")
    print("   - Kernel structure correct")
    print("   - NeuralBridge integration verified")
else:
    print("⚠️ Overall Status: VALIDATION ISSUES DETECTED")
    if not all_operators_present:
        print("   - Missing MF operators")
    if not kernel_ok:
        print("   - Kernel structure issues")
    if not bridge_integration:
        print("   - Bridge integration issues")

print()

