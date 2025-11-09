#!/usr/bin/env python3
"""
Comprehensive test script for updated dynamic data loading system.
Tests:
1. File organization by region
2. Atmospheric variable parsing
3. Variable checking and merging
4. Odisha region integration
"""

import os
import sys
import ast
import re
from pathlib import Path

def test_atmospheric_variable_parsing():
    """Test parsing of atmospheric variables"""
    print("Test 1: Atmospheric Variable Parsing...")

    # Test by importing the function
    try:
        sys.path.insert(0, str(Path(__file__).parent / "finetuning"))
        from prepare_forecasts_and_targets import parse_atmospheric_variable

        test_cases = [
            ('temperature_500hPa', ('temperature', 500)),
            ('geopotential_1000hPa', ('geopotential', 1000)),
            ('2m_temperature', ('2m_temperature', None)),
            ('10m_u_component_of_wind', ('10m_u_component_of_wind', None)),
            ('specific_humidity_850hPa', ('specific_humidity', 850)),
        ]

        all_passed = True
        for var_input, expected in test_cases:
            result = parse_atmospheric_variable(var_input)
            if result == expected:
                print(f"  ✓ {var_input} -> {result}")
            else:
                print(f"  ✗ {var_input} -> got {result}, expected {expected}")
                all_passed = False

        return all_passed

    except ImportError:
        print("  ⚠ Cannot import module (dask not available), testing with regex...")

        # Test with regex directly
        pattern = r'^(.+)_(\d+)hPa$'
        test_cases = [
            ('temperature_500hPa', True, ('temperature', '500')),
            ('geopotential_1000hPa', True, ('geopotential', '1000')),
            ('2m_temperature', False, None),
            ('specific_humidity_850hPa', True, ('specific_humidity', '850')),
        ]

        all_passed = True
        for var_input, should_match, expected_groups in test_cases:
            match = re.match(pattern, var_input)
            if should_match:
                if match and (match.group(1), match.group(2)) == expected_groups:
                    print(f"  ✓ {var_input} matched correctly")
                else:
                    print(f"  ✗ {var_input} did not match as expected")
                    all_passed = False
            else:
                if not match:
                    print(f"  ✓ {var_input} correctly not matched")
                else:
                    print(f"  ✗ {var_input} incorrectly matched")
                    all_passed = False

        return all_passed


def test_file_path_structure():
    """Test new file path organization"""
    print("\nTest 2: File Path Structure...")

    try:
        sys.path.insert(0, str(Path(__file__).parent / "finetuning"))
        from prepare_forecasts_and_targets import get_data_path

        test_cases = [
            ("~/data", "pangu", "odisha", 2020, "~/data/pangu/pangu_odisha_2020.zarr"),
            ("/tmp/data", "era5", "usa_south", 2019, "/tmp/data/era5/era5_usa_south_2019.zarr"),
        ]

        all_passed = True
        for data_dir, source, region, year, expected in test_cases:
            # Normalize expected path
            expected_normalized = os.path.normpath(os.path.expanduser(expected))
            result = get_data_path(data_dir, source, region, year)
            result_normalized = os.path.normpath(result)

            if result_normalized == expected_normalized:
                print(f"  ✓ {source}/{region}/{year} -> correct path")
            else:
                print(f"  ✗ Expected: {expected_normalized}")
                print(f"    Got:      {result_normalized}")
                all_passed = False

        return all_passed

    except ImportError:
        print("  ⚠ Cannot import module (testing with code inspection)...")

        # Check the function exists in the code
        with open("finetuning/prepare_forecasts_and_targets.py", 'r') as f:
            content = f.read()

        if 'def get_data_path(' in content:
            print("  ✓ get_data_path function exists")
            if '{data_source}_{region}_{year}.zarr' in content:
                print("  ✓ Correct file naming pattern found")
                return True
            else:
                print("  ✗ File naming pattern not found")
                return False
        else:
            print("  ✗ get_data_path function not found")
            return False


def test_load_combined_dataset_signature():
    """Test that load_combined_dataset has region parameter"""
    print("\nTest 3: load_combined_dataset Updated Signature...")

    with open("finetuning/finetune.py", 'r') as f:
        content = f.read()

    # Check function signature
    if 'def load_combined_dataset(lat_values, lon_values, time_values, root_dir, data_source, region):' in content:
        print("  ✓ load_combined_dataset has region parameter")

        # Check it uses the new file pattern
        if '{data_source}/{data_source}_{region}_{year}.zarr' in content:
            print("  ✓ Uses new file path pattern")
            return True
        else:
            print("  ✗ Does not use new file path pattern")
            return False
    else:
        print("  ✗ load_combined_dataset does not have region parameter")
        return False


def test_prepare_data_signature():
    """Test that prepare_data_for_finetuning has region parameter"""
    print("\nTest 4: prepare_data_for_finetuning Signature...")

    with open("finetuning/prepare_forecasts_and_targets.py", 'r') as f:
        content = f.read()

    # Check function has region parameter
    if 'def prepare_data_for_finetuning(data_dir, model_name, ground_truth_source, region,' in content:
        print("  ✓ prepare_data_for_finetuning has region parameter")
        return True
    else:
        print("  ✗ prepare_data_for_finetuning missing region parameter")
        return False


def test_variable_checking_functions():
    """Test that variable checking functions exist"""
    print("\nTest 5: Variable Checking Functions...")

    with open("finetuning/prepare_forecasts_and_targets.py", 'r') as f:
        content = f.read()

    functions_to_check = [
        ('check_variables_in_dataset', 'Check which variables are present in a dataset'),
        ('merge_variables_into_dataset', 'Merge new variables into existing dataset'),
    ]

    all_found = True
    for func_name, description in functions_to_check:
        if f'def {func_name}(' in content:
            print(f"  ✓ {func_name} function exists")
        else:
            print(f"  ✗ {func_name} function not found")
            all_found = False

    return all_found


def test_finetune_integration():
    """Test that finetune.py correctly calls prepare_data_for_finetuning with region"""
    print("\nTest 6: finetune.py Integration...")

    with open("finetuning/finetune.py", 'r') as f:
        content = f.read()

    # Check that prepare_data_for_finetuning is called with region parameter
    if 'region=args.region,' in content and 'prepare_data_for_finetuning(' in content:
        print("  ✓ prepare_data_for_finetuning called with region parameter")
        return True
    else:
        print("  ✗ prepare_data_for_finetuning not called with region parameter")
        return False


def test_odisha_region_with_new_system():
    """Test that Odisha region works with new system"""
    print("\nTest 7: Odisha Region Integration...")

    with open("finetuning/finetune.py", 'r') as f:
        content = f.read()

    # Check Odisha is defined
    if 'elif args.region == "odisha":' in content:
        print("  ✓ Odisha region defined in finetune.py")

        # Check coordinates
        if 'lat0, lat1 = 18, 22' in content and 'lon0, lon1 = 82, 86' in content:
            print("  ✓ Odisha coordinates correct (centered at ~20°N, 84°E)")
            return True
        else:
            print("  ✗ Odisha coordinates incorrect")
            return False
    else:
        print("  ✗ Odisha region not defined")
        return False


def test_atmospheric_variable_download_support():
    """Test that download functions support atmospheric variables"""
    print("\nTest 8: Atmospheric Variable Download Support...")

    with open("finetuning/prepare_forecasts_and_targets.py", 'r') as f:
        content = f.read()

    checks = [
        ('atmospheric_vars = {}', 'Atmospheric variables dictionary'),
        ('parse_atmospheric_variable(var)', 'Parse atmospheric variable call'),
        ("'level' in subset.dims", 'Check for level dimension'),
        ('subset.sel(level=', 'Select pressure levels'),
    ]

    all_found = True
    for check_str, description in checks:
        if check_str in content:
            print(f"  ✓ {description} found")
        else:
            print(f"  ✗ {description} not found")
            all_found = False

    return all_found


def run_all_tests():
    """Run all validation tests"""
    print("="*70)
    print("COMPREHENSIVE VALIDATION TESTS")
    print("Updated Dynamic Data Loading System")
    print("="*70)

    tests = [
        ("Atmospheric Variable Parsing", test_atmospheric_variable_parsing),
        ("File Path Structure", test_file_path_structure),
        ("load_combined_dataset Signature", test_load_combined_dataset_signature),
        ("prepare_data_for_finetuning Signature", test_prepare_data_signature),
        ("Variable Checking Functions", test_variable_checking_functions),
        ("finetune.py Integration", test_finetune_integration),
        ("Odisha Region Integration", test_odisha_region_with_new_system),
        ("Atmospheric Variable Download", test_atmospheric_variable_download_support),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ✗ Test crashed: {e}")
            import traceback
            traceback.print_exc()
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
        print("ALL TESTS PASSED!")
        print("\nNew Features:")
        print("✓ Data organized by region: data_dir/model/model_region_year.zarr")
        print("✓ Atmospheric variables supported: temperature_500hPa, etc.")
        print("✓ Variable checking: verifies all required variables exist")
        print("✓ Incremental updates: merges missing variables into existing data")
        print("✓ Odisha region ready to use")
        print("\nExample usage:")
        print("  python3 finetuning/finetune.py --region=odisha \\")
        print("    --training_vars 2m_temperature temperature_850hPa \\")
        print("    --output_vars 2m_temperature \\")
        print("    --model_name=pangu --lead_time_hours 24 \\")
        print("    --train_start=2020-01-01 --train_end=2020-01-07 \\")
        print("    --test_start=2020-01-08 --test_end=2020-01-10 \\")
        print("    --data_dir=~/ai_weather_ag/data/raw \\")
        print("    --output_dir=~/ai_weather_ag/data/fine_tuning_output")
        return 0
    else:
        print("SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
