import argparse
import sys
from pathlib import Path
import torch

try:
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError:
    print("Error: safetensors library not found. Install with: pip install safetensors")
    sys.exit(1)


def load_safetensors_dict(file_path):
    """Load all tensors from a safetensors file into a dictionary"""
    tensors = {}
    try:
        with safe_open(file_path, framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        return tensors
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def get_safetensors_keys(file_path):
    """Get keys from a safetensors file"""
    try:
        with safe_open(file_path, framework="pt") as f:
            return set(f.keys())
    except Exception as e:
        print(f"Error reading keys from {file_path}: {e}")
        return None


def merge_safetensors(source_file, target_file, output_file):
    """
    Merge two safetensors files by copying missing keys from source to target

    Args:
        source_file: Path to source file (file with more keys)
        target_file: Path to target file (file with fewer keys)
        output_file: Path to output merged file
    """
    print(f"Loading source file: {source_file}")
    source_tensors = load_safetensors_dict(source_file)
    if source_tensors is None:
        return False

    print(f"Loading target file: {target_file}")
    target_tensors = load_safetensors_dict(target_file)
    if target_tensors is None:
        return False

    print(f"Source file has {len(source_tensors)} keys")
    print(f"Target file has {len(target_tensors)} keys")

    # Find missing keys
    source_keys = set(source_tensors.keys())
    target_keys = set(target_tensors.keys())
    missing_keys = source_keys - target_keys

    print(f"Found {len(missing_keys)} missing keys in target file")

    if len(missing_keys) == 0:
        print("No missing keys found. Files already have the same keys.")
        return True

    # Create merged dictionary starting with target tensors
    merged_tensors = target_tensors.copy()

    # Add missing keys from source
    print("Copying missing keys from source to target...")
    for key in missing_keys:
        merged_tensors[key] = source_tensors[key]
        if len(missing_keys) <= 20:  # Only show individual keys if not too many
            print(f"  Added: {key}")

    if len(missing_keys) > 20:
        print(f"  (Added {len(missing_keys)} keys total)")

    print(f"Merged file will have {len(merged_tensors)} keys")

    # Convert to bfloat16 before saving
    print("Converting floating-point tensors to bfloat16...")
    for key, tensor in merged_tensors.items():
        if tensor.is_floating_point():
            merged_tensors[key] = tensor.to(torch.bfloat16)

    # Save merged file
    print(f"Saving merged file to: {output_file}")
    try:
        save_file(merged_tensors, output_file)
        print("Successfully saved merged file!")

        # Verify the merge
        print("\nVerifying merged file...")
        merged_keys = get_safetensors_keys(output_file)
        if merged_keys and len(merged_keys) == len(source_keys):
            print("✓ Verification successful: All keys present in merged file")
            return True
        else:
            print("✗ Verification failed: Key count mismatch")
            return False

    except Exception as e:
        print(f"Error saving merged file: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Merge safetensors files by copying missing keys from source to target"
    )
    parser.add_argument(
        "--source_file", help="Source safetensors file (with more keys)"
    )
    parser.add_argument(
        "--target_file", help="Target safetensors file (with fewer keys)"
    )
    parser.add_argument("--output_file", help="Output merged safetensors file")
    parser.add_argument(
        "--force", action="store_true", help="Overwrite output file if it exists"
    )

    args = parser.parse_args()

    # Validate input files
    source_path = Path(args.source_file)
    target_path = Path(args.target_file)
    output_path = Path(args.output_file)

    if not source_path.exists():
        print(f"Error: Source file not found: {source_path}")
        sys.exit(1)

    if not target_path.exists():
        print(f"Error: Target file not found: {target_path}")
        sys.exit(1)

    if output_path.exists() and not args.force:
        print(f"Error: Output file already exists: {output_path}")
        print("Use --force to overwrite")
        sys.exit(1)

    # Perform the merge
    success = merge_safetensors(source_path, target_path, output_path)

    if success:
        print("\n" + "=" * 60)
        print("MERGE COMPLETED SUCCESSFULLY!")
        print(f"Merged file saved as: {output_path}")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("MERGE FAILED!")
        print("=" * 60)
        sys.exit(1)


# Convenience function for programmatic use
def merge_safetensors_files(source_file, target_file, output_file):
    """
    Simple function to merge two safetensors files

    Args:
        source_file: Path to source file (with more keys)
        target_file: Path to target file (with fewer keys)
        output_file: Path to output merged file

    Returns:
        bool: True if successful, False otherwise
    """
    return merge_safetensors(source_file, target_file, output_file)


if __name__ == "__main__":
    main()
