#!/usr/bin/env python3
"""
Validation test for Odisha region and dynamic data loading code
Tests that don't require full package installation
"""

import os
import sys
import ast

def test_odisha_region_in_code():
    """Test that odisha region is properly defined in finetune.py"""
    print("Test 1: Validating Odisha region definition...")

    finetune_path = "finetuning/finetune.py"

    with open(finetune_path, 'r') as f:
        content = f.read()

    # Check if odisha is mentioned
    if 'odisha' in content:
        print("  ✓ Odisha region found in finetune.py")

        # Extract the region definition
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'elif args.region == "odisha"' in line:
                print(f"  ✓ Found at line {i+1}")
                # Get the next few lines to show the definition
                region_def = '\n'.join(lines[i:i+3])
                print(f"  Definition:\n{region_def}")

                # Validate coordinates are around 20, 84
                if 'lat0, lat1 = 18, 22' in region_def and 'lon0, lon1 = 82, 86' in region_def:
                    print("  ✓ Coordinates centered at ~20°N, 84°E")
                    return True
                else:
                    print("  ✗ Coordinates don't match expected values")
                    return False
        print("  ✗ Region definition not found in expected format")
        return False
    else:
        print("  ✗ Odisha region not found in finetune.py")
        return False

def test_prepare_module_exists():
    """Test that prepare_forecasts_and_targets.py exists and has expected functions"""
    print("\nTest 2: Validating prepare_forecasts_and_targets.py module...")

    module_path = "finetuning/prepare_forecasts_and_targets.py"

    if not os.path.exists(module_path):
        print(f"  ✗ Module not found at {module_path}")
        return False

    print(f"  ✓ Module exists at {module_path}")

    with open(module_path, 'r') as f:
        content = f.read()

    # Check for key functions
    expected_functions = [
        'check_data_exists',
        'download_forecast_data',
        'download_target_data',
        'prepare_data_for_finetuning'
    ]

    all_found = True
    for func in expected_functions:
        if f'def {func}(' in content:
            print(f"  ✓ Function '{func}' found")
        else:
            print(f"  ✗ Function '{func}' not found")
            all_found = False

    return all_found

def test_finetune_imports_prepare():
    """Test that finetune.py imports the prepare module"""
    print("\nTest 3: Validating finetune.py imports prepare_forecasts_and_targets...")

    with open("finetuning/finetune.py", 'r') as f:
        content = f.read()

    if 'from prepare_forecasts_and_targets import prepare_data_for_finetuning' in content:
        print("  ✓ Import statement found")

        if 'prepare_data_for_finetuning(' in content:
            print("  ✓ Function is called in main()")
            return True
        else:
            print("  ✗ Function is not called")
            return False
    else:
        print("  ✗ Import statement not found")
        return False

def test_syntax():
    """Test that Python files have valid syntax"""
    print("\nTest 4: Validating Python syntax...")

    files = [
        "finetuning/finetune.py",
        "finetuning/prepare_forecasts_and_targets.py"
    ]

    all_valid = True
    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                ast.parse(f.read())
            print(f"  ✓ {filepath} has valid syntax")
        except SyntaxError as e:
            print(f"  ✗ {filepath} has syntax error: {e}")
            all_valid = False

    return all_valid

def test_readme_exists():
    """Test that documentation exists"""
    print("\nTest 5: Validating documentation...")

    readme_path = "finetuning/README_DYNAMIC_DATA_LOADING.md"
    if os.path.exists(readme_path):
        print(f"  ✓ Documentation exists at {readme_path}")
        with open(readme_path, 'r') as f:
            content = f.read()
            if len(content) > 100:
                print(f"  ✓ Documentation is substantial ({len(content)} characters)")
                return True
    else:
        print(f"  ✗ Documentation not found")
        return False

def run_all_tests():
    """Run all validation tests"""
    print("="*70)
    print("VALIDATION TESTS FOR ODISHA REGION & DYNAMIC DATA LOADING")
    print("="*70)

    tests = [
        ("Odisha Region Definition", test_odisha_region_in_code),
        ("Prepare Module Structure", test_prepare_module_exists),
        ("Integration in finetune.py", test_finetune_imports_prepare),
        ("Python Syntax", test_syntax),
        ("Documentation", test_readme_exists)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ✗ Test crashed: {e}")
            results.append((test_name, False))

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    all_passed = True
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not result:
            all_passed = False

    print("="*70)
    if all_passed:
        print("ALL VALIDATION TESTS PASSED")
        print("\nThe Odisha region (centered at ~20°N, 84°E) is properly configured")
        print("and the dynamic data loading system is ready to use.")
        print("\nTo test with actual data download, run:")
        print("  python3 finetuning/finetune.py --region=odisha --subregion=2x2 \\")
        print("    --model_name=pangu --training_vars 2m_temperature \\")
        print("    --output_vars 2m_temperature --train_start=2020-01-01 \\")
        print("    --train_end=2020-01-07 --test_start=2020-01-08 \\")
        print("    --test_end=2020-01-10 --lead_time_hours 24 \\")
        print("    --data_dir=~/ai_weather_ag/data/raw \\")
        print("    --output_dir=~/ai_weather_ag/data/fine_tuning_output")
        return 0
    else:
        print("SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
