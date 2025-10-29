#!/usr/bin/env python3
"""
Script to recursively convert Zarr v2 stores to Zarr v3 format.

This script walks through a directory tree, identifies Zarr v2 stores,
and converts them to Zarr v3 format in-place.
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path
import zarr
import xarray as xr


def detect_zarr_version(zarr_path):
    """
    Detect whether a zarr store is v2 or v3.
    
    Parameters
    ----------
    zarr_path : str or Path
        Path to the zarr store
    
    Returns
    -------
    int or None
        2 for Zarr v2, 3 for Zarr v3, None if not a valid zarr store
    """
    zarr_path = Path(zarr_path)
    
    # Check for Zarr v2 indicators
    zarray_v2 = zarr_path / '.zarray'
    zgroup_v2 = zarr_path / '.zgroup'
    
    # Check for Zarr v3 indicators
    zarr_json = zarr_path / 'zarr.json'
    
    if zarr_json.exists():
        return 3
    elif zarray_v2.exists() or zgroup_v2.exists():
        return 2
    else:
        return None


def copy_zarr_v2_to_v3_xarray(src_path, dst_path):
    """
    Copy a Zarr v2 store to v3 format using xarray.

    Parameters
    ----------
    src_path : str or Path
        Path to source Zarr v2 store
    dst_path : str or Path
        Path to destination Zarr v3 store
    """
    print(f"    Loading data from v2 store...")

    # Open the v2 store with xarray
    ds = xr.open_zarr(src_path, zarr_format=2)

    print(f"    Writing to v3 store...")

    # Build encoding to handle codec conversion from v2 to v3
    # Remove the old compressor settings and let zarr v3 use defaults
    encoding = {}
    for var in ds.variables:
        encoding[var] = {
            'compressor': None,  # Remove v2 compressor
            'chunks': ds[var].encoding.get('chunks', None),
        }

    # Write to v3 format with the cleaned encoding
    ds.to_zarr(dst_path, zarr_format=3, encoding=encoding, consolidated=False)

    # Close the dataset
    ds.close()


def convert_zarr_v2_to_v3(zarr_path, backup=True):
    """
    Convert a Zarr v2 store to v3 format using xarray.

    Parameters
    ----------
    zarr_path : str or Path
        Path to the zarr v2 store
    backup : bool, optional
        Whether to create a backup before conversion (default: True)

    Returns
    -------
    bool
        True if conversion successful, False otherwise
    """
    zarr_path = Path(zarr_path)

    try:
        print(f"Converting {zarr_path} from Zarr v2 to v3...")

        # # Create backup if requested
        # if backup:
        #     backup_path = zarr_path.with_suffix('.zarr.v2_backup')
        #     if backup_path.exists():
        #         print(f"  Warning: Backup {backup_path} already exists, skipping backup creation")
        #     else:
        #         print(f"  Creating backup at {backup_path}")
        #         shutil.copytree(zarr_path, backup_path)

        # Create temporary directory for v3 store
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / 'temp.zarr'

            # Use xarray to handle the conversion
            copy_zarr_v2_to_v3_xarray(zarr_path, temp_path)

            # Remove original v2 store
            shutil.rmtree(zarr_path)

            # Move v3 store to original location
            shutil.move(str(temp_path), str(zarr_path))

        print(f"  ✓ Successfully converted {zarr_path}")
        return True

    except Exception as e:
        print(f"  ✗ Error converting {zarr_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_conversion(zarr_path):
    """
    Verify that a converted Zarr store can be opened and read.
    
    Parameters
    ----------
    zarr_path : str or Path
        Path to the zarr store
    
    Returns
    -------
    bool
        True if verification successful, False otherwise
    """
    try:
        store = zarr.open_group(str(zarr_path), mode='r')
        # Try to access basic metadata
        _ = store.info
        version = detect_zarr_version(zarr_path)
        print(f"  ✓ Verification: Store opens successfully as Zarr v{version}")
        return True
    except Exception as e:
        print(f"  ✗ Verification failed: {e}")
        return False


def process_directory(root_dir, backup=True, dry_run=False, test_first=False):
    """
    Recursively process a directory and convert all Zarr v2 stores to v3.
    
    Parameters
    ----------
    root_dir : str or Path
        Root directory to process
    backup : bool, optional
        Whether to create backups before conversion (default: True)
    dry_run : bool, optional
        If True, only identify files without converting (default: False)
    test_first : bool, optional
        If True, convert only the first v2 file and wait for confirmation (default: False)
    
    Returns
    -------
    dict
        Summary statistics of the conversion process
    """
    root_dir = Path(root_dir)
    
    if not root_dir.exists():
        raise ValueError(f"Directory does not exist: {root_dir}")
    
    stats = {
        'total_zarr': 0,
        'v2_found': 0,
        'v3_found': 0,
        'converted': 0,
        'failed': 0
    }
    
    # Find all .zarr directories
    zarr_stores = []
    for item in root_dir.rglob('*.zarr'):
        if item.is_dir():
            zarr_stores.append(item)
    
    print(f"Found {len(zarr_stores)} .zarr directories in {root_dir}")
    print("-" * 70)
    
    # Track if we're in test mode and have converted the first file
    test_mode_converted = False
    
    for zarr_path in sorted(zarr_stores):
        stats['total_zarr'] += 1
        version = detect_zarr_version(zarr_path)
        
        if version == 2:
            stats['v2_found'] += 1
            print(f"\n[Zarr v2] {zarr_path}")
            
            if dry_run:
                print("  (Dry run - would convert)")
            else:
                # In test mode, skip if we've already handled the first file
                if test_first and test_mode_converted:
                    print("  (Skipping - test mode, already processed first file)")
                    continue

                success = convert_zarr_v2_to_v3(zarr_path, backup=backup)
                if success:
                    stats['converted'] += 1

                    # Verify the conversion
                    verify_conversion(zarr_path)
                else:
                    stats['failed'] += 1

                # If in test mode and this was the first v2 file (success or fail), ask for confirmation
                if test_first and not test_mode_converted:
                    test_mode_converted = True
                    print("\n" + "=" * 70)
                    print("TEST CONVERSION COMPLETE")
                    print("=" * 70)
                    if success:
                        print(f"✓ Successfully converted first file: {zarr_path}")
                        print(f"  Original backup is at: {zarr_path.with_suffix('.zarr.v2_backup')}")
                    else:
                        print(f"✗ Failed to convert first file: {zarr_path}")
                        print(f"  Please review the error above before proceeding.")
                    print(f"\nFound {stats['v2_found'] - 1} more Zarr v2 files to convert.")
                    print("-" * 70)

                    response = input("\nContinue with remaining conversions? (yes/no): ")
                    if response.lower() != 'yes':
                        print("\nConversion stopped by user.")
                        return stats
                    print("\nProceeding with remaining conversions...\n")
                    print("=" * 70)
                    
        elif version == 3:
            stats['v3_found'] += 1
            print(f"\n[Zarr v3] {zarr_path} - already v3, skipping")
            
        else:
            print(f"\n[Unknown] {zarr_path} - could not determine version")
    
    return stats


def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Recursively convert Zarr v2 stores to v3 format'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create backups before conversion'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only identify files without converting'
    )
    parser.add_argument(
        '--test-first',
        action='store_true',
        help='Convert only the first v2 file and wait for confirmation before proceeding'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Zarr v2 to v3 Conversion Script")
    print("=" * 70)
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
    
    if args.no_backup and not args.dry_run:
        print("WARNING: Running without backups!")
        response = input("Are you sure? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return
    
    print()
    
    try:
        stats = process_directory(
            root_dir = "/Users/ohouck/globus/forecast_data",
            backup=not args.no_backup,
            dry_run=args.dry_run,
            test_first=args.test_first
        )
        
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total .zarr directories found: {stats['total_zarr']}")
        print(f"Zarr v2 stores found: {stats['v2_found']}")
        print(f"Zarr v3 stores found: {stats['v3_found']}")
        
        if not args.dry_run:
            print(f"Successfully converted: {stats['converted']}")
            print(f"Failed conversions: {stats['failed']}")
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()