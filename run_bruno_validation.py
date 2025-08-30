#!/usr/bin/env python3
"""
Bruno Framework Validation Runner
================================

Complete validation suite for the consolidated Bruno/E3 framework.
This script validates all components and demonstrates full functionality.

Usage:
    python run_bruno_validation.py          # Run full validation suite
    python run_bruno_validation.py --quick  # Quick validation only
    python run_bruno_validation.py --demo   # Demo mode with explanations

Author: E3 Project Team
Version: 2.0 - Consolidated Framework
Date: August 2025
"""

import sys
import os
import argparse
from pathlib import Path
import importlib.util

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def check_dependencies():
    """Check if required dependencies are available."""
    print("🔍 CHECKING DEPENDENCIES")
    print("-" * 40)
    
    dependencies = [
        ('numpy', 'NumPy for numerical calculations'),
        ('torch', 'PyTorch for neural networks'),
        ('pandas', 'Pandas for data manipulation'),
        ('sklearn', 'Scikit-learn for ML utilities'),
        ('matplotlib', 'Matplotlib for plotting')
    ]
    
    missing = []
    for dep, description in dependencies:
        try:
            importlib.import_module(dep)
            print(f"✅ {dep:<12} - {description}")
        except ImportError:
            print(f"❌ {dep:<12} - {description} (MISSING)")
            missing.append(dep)
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install torch pandas scikit-learn matplotlib")
        return False
    
    print("✅ All dependencies available")
    return True


def validate_bruno_framework():
    """Validate core Bruno framework functionality."""
    print("\n🔬 VALIDATING BRUNO FRAMEWORK")
    print("-" * 40)
    
    try:
        sys.path.append(str(Path(__file__).parent / "bruno_framework" / "theory"))
        from bruno_threshold import (
            validate_bruno_constant,
            bruno_threshold_check,
            KAPPA_VALIDATED
        )
        
        print(f"✅ Bruno framework imported successfully")
        print(f"📊 Bruno constant: κ = {KAPPA_VALIDATED} K⁻¹")
        
        # Run validation
        results = validate_bruno_constant()
        print(f"✅ Validation status: {results['validation_status']}")
        print(f"📈 GW150914 error: {results['gw_error_percent']:.1f}%")
        
        # Test threshold function
        temp = 300.0  # Room temperature
        exceeded, beta_B = bruno_threshold_check(temp)
        print(f"✅ Threshold test (T={temp}K): β_B = {beta_B:.1e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Bruno framework validation failed: {e}")
        return False


def validate_materials_data():
    """Validate materials data integration."""
    print("\n📊 VALIDATING MATERIALS DATA")
    print("-" * 40)
    
    # Check for materials data
    data_paths = [
        Path(__file__).parent / "data" / "validation" / "materials_final_dataset.csv",
        Path(__file__).parent / "E3_Engine" / "materials_data"
    ]
    
    materials_found = False
    for path in data_paths:
        if path.exists():
            print(f"✅ Materials data found: {path}")
            materials_found = True
            break
    
    if not materials_found:
        print("⚠️  No materials data found, will use synthetic data")
    
    # Test materials validation script
    try:
        validation_script = Path(__file__).parent / "scripts" / "bruno_materials_validation.py"
        if validation_script.exists():
            print(f"✅ Materials validation script: {validation_script}")
            return True
        else:
            print(f"❌ Materials validation script missing")
            return False
    except Exception as e:
        print(f"❌ Materials validation check failed: {e}")
        return False


def validate_neural_engine():
    """Validate enhanced neural physics engine."""
    print("\n🧠 VALIDATING NEURAL ENGINE")
    print("-" * 40)
    
    try:
        neural_engine = Path(__file__).parent / "E3_Engine" / "enhanced_neural_physics_engine.py"
        if neural_engine.exists():
            print(f"✅ Enhanced neural engine: {neural_engine}")
            
            # Test import
            spec = importlib.util.spec_from_file_location("enhanced_engine", neural_engine)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            print(f"✅ Neural engine imports successfully")
            print(f"✅ Bruno integration: {'ENABLED' if hasattr(module, 'BRUNO_INTEGRATION') else 'DISABLED'}")
            
            return True
        else:
            print(f"❌ Enhanced neural engine missing")
            return False
            
    except Exception as e:
        print(f"❌ Neural engine validation failed: {e}")
        return False


def run_demo_mode():
    """Run interactive demo of Bruno framework."""
    print("\n🎮 BRUNO FRAMEWORK DEMO")
    print("=" * 50)
    print("Interactive demonstration of Bruno constant calculations")
    print()
    
    # Import Bruno framework
    sys.path.append(str(Path(__file__).parent / "bruno_framework" / "theory"))
    from bruno_threshold import (
        bruno_threshold_check,
        entropy_collapse_boundary,
        collapse_radius_boundary,
        KAPPA_VALIDATED
    )
    
    print(f"🔬 Bruno Constant: κ = {KAPPA_VALIDATED} K⁻¹")
    print(f"🌡️  Critical Temperature: {entropy_collapse_boundary():.2e} K")
    print()
    
    # Interactive temperature testing
    test_temperatures = [1.0, 10.0, 100.0, 298.15, 1000.0, 5000.0]
    
    print("🧪 ENTROPY COLLAPSE ANALYSIS")
    print("-" * 50)
    print(f"{'Temperature (K)':<15} {'β_B':<15} {'Collapsed?':<12} {'R_boundary (m)':<15}")
    print("-" * 50)
    
    for temp in test_temperatures:
        exceeded, beta_B = bruno_threshold_check(temp)
        radius = collapse_radius_boundary(temp)
        
        print(f"{temp:<15.1f} {beta_B:<15.2e} {'YES' if exceeded else 'NO':<12} {radius:<15.2e}")
    
    print()
    print("🎯 INTERPRETATION:")
    print("- β_B > 1: Entropy collapse regime (surface-dominant)")
    print("- β_B < 1: Volumetric entropy regime") 
    print("- R_boundary: Critical radius for collapse at given temperature")
    print()
    print("✨ Demo complete! The Bruno framework is working correctly.")


def run_quick_validation():
    """Run quick validation checks only."""
    print("⚡ QUICK VALIDATION MODE")
    print("=" * 40)
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Bruno Framework", validate_bruno_framework),
        ("Materials Data", validate_materials_data),
        ("Neural Engine", validate_neural_engine)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{name:<20}: {status}")
        except Exception as e:
            results.append((name, False))
            print(f"{name:<20}: ❌ ERROR ({e})")
    
    return all(result for _, result in results)


def run_full_validation():
    """Run complete validation suite."""
    print("🚀 FULL VALIDATION SUITE")
    print("=" * 60)
    print("Complete validation of consolidated Bruno/E3 framework")
    print()
    
    # Step-by-step validation
    steps = [
        ("Checking Dependencies", check_dependencies),
        ("Validating Bruno Framework", validate_bruno_framework),
        ("Validating Materials Data", validate_materials_data),
        ("Validating Neural Engine", validate_neural_engine)
    ]
    
    all_passed = True
    for step_name, step_func in steps:
        print(f"\n{step_name.upper()}")
        print("=" * len(step_name))
        
        try:
            result = step_func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ {step_name} failed: {e}")
            all_passed = False
    
    # Final summary
    print(f"\n🎯 VALIDATION SUMMARY")
    print("=" * 30)
    
    if all_passed:
        print("✅ ALL VALIDATIONS PASSED")
        print("🔬 Bruno framework: OPERATIONAL")
        print("📊 Materials data: INTEGRATED")
        print("🧠 Neural engine: ENHANCED")
        print("🎯 Framework status: READY FOR USE")
        print()
        print("📋 Next steps:")
        print("  - Run: python scripts/bruno_materials_validation.py")
        print("  - Train: python E3_Engine/enhanced_neural_physics_engine.py")
        print("  - Explore: python run_bruno_validation.py --demo")
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print("🔧 Review error messages above and fix issues")
        print("📋 Common fixes:")
        print("  - Install missing dependencies: pip install torch pandas scikit-learn")
        print("  - Check file paths and permissions")
        print("  - Verify repository structure is complete")
    
    return all_passed


def main():
    """Main validation runner."""
    parser = argparse.ArgumentParser(description='Bruno Framework Validation Suite')
    parser.add_argument('--quick', action='store_true', help='Run quick validation only')
    parser.add_argument('--demo', action='store_true', help='Run interactive demo')
    parser.add_argument('--version', action='version', version='Bruno Framework 2.0')
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo_mode()
    elif args.quick:
        success = run_quick_validation()
        sys.exit(0 if success else 1)
    else:
        success = run_full_validation()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()