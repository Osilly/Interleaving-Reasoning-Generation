#!/usr/bin/env python3
"""
EMA Weight Conversion Script: Convert EMA weights from float32 to bfloat16 format

Usage:
    python convert_ema_to_bf16.py --input_path /path/to/checkpoint/ema.safetensors --output_path /path/to/output/ema_bf16.safetensors

    Or specify checkpoint directory (automatically find ema.safetensors):
    python convert_ema_to_bf16.py --checkpoint_dir /path/to/checkpoint --output_path /path/to/output/ema_bf16.safetensors
"""

import argparse
import os
import sys
from pathlib import Path
import torch

try:
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError:
    print("Error: safetensors library not found. Please install: pip install safetensors")
    sys.exit(1)


def load_ema_weights(file_path):
    """
    Load EMA weights from safetensors file

    Args:
        file_path: Path to EMA weights file

    Returns:
        dict: Dictionary containing all weight tensors, returns None if failed
    """
    print(f"Loading EMA weights file: {file_path}")

    if not os.path.exists(file_path):
        print(f"Error: File does not exist: {file_path}")
        return None

    try:
        tensors = {}
        with safe_open(file_path, framework="pt") as f:
            keys = list(f.keys())
            print(f"Found {len(keys)} weight tensors")

            for key in keys:
                tensor = f.get_tensor(key)
                tensors[key] = tensor

        return tensors

    except Exception as e:
        print(f"Error loading file: {e}")
        return None


def convert_to_bf16(tensors):
    """
    Convert weight tensors to bfloat16 format

    Args:
        tensors: Dictionary of weight tensors

    Returns:
        dict: Dictionary of converted weight tensors
    """
    print("Converting weights to bfloat16 format...")

    converted_tensors = {}
    float_count = 0
    other_count = 0

    for key, tensor in tensors.items():
        if tensor.is_floating_point():
            # Convert floating point tensors to bf16
            converted_tensors[key] = tensor.to(torch.bfloat16)
            float_count += 1
        else:
            # Keep non-floating point tensors unchanged (like integer indices)
            converted_tensors[key] = tensor
            other_count += 1

    print(
        f"Conversion completed: {float_count} floating point tensors converted to bf16, {other_count} other type tensors kept unchanged"
    )
    return converted_tensors


def save_bf16_weights(tensors, output_path):
    """
    Save bf16 format weights

    Args:
        tensors: Dictionary of weight tensors
        output_path: Output file path

    Returns:
        bool: Whether saving was successful
    """
    print(f"Saving bf16 weights to: {output_path}")

    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Save weights
        save_file(tensors, output_path)
        print("Save successful!")

        # Verify saved file
        print("Verifying saved file...")
        with safe_open(output_path, framework="pt") as f:
            saved_keys = set(f.keys())
            original_keys = set(tensors.keys())

            if saved_keys == original_keys:
                print("✓ Verification successful: All weights saved correctly")

                # Check data type
                sample_key = list(saved_keys)[0]
                sample_tensor = f.get_tensor(sample_key)
                if sample_tensor.is_floating_point():
                    print(f"✓ Data type verification: {sample_tensor.dtype}")

                return True
            else:
                print("✗ Verification failed: Weight keys don't match")
                return False

    except Exception as e:
        print(f"Error saving file: {e}")
        return False


def get_file_size_mb(file_path):
    """Get file size in MB"""
    if os.path.exists(file_path):
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Convert EMA weights from float32 to bfloat16 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Directly specify EMA file path
  python convert_ema_to_bf16.py --input_path /path/to/ema.safetensors --output_path /path/to/ema_bf16.safetensors
  
  # Specify checkpoint directory (automatically find ema.safetensors)
  python convert_ema_to_bf16.py --checkpoint_dir /path/to/checkpoint --output_path /path/to/ema_bf16.safetensors
  
  # Use default output filename
  python convert_ema_to_bf16.py --input_path /path/to/ema.safetensors
        """,
    )

    # Input options (choose one)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_path", help="Full path to EMA weights file (ema.safetensors)"
    )
    input_group.add_argument(
        "--checkpoint_dir", help="Checkpoint directory path (will automatically find ema.safetensors file in it)"
    )

    # Output options
    parser.add_argument(
        "--output_path",
        help="Output bf16 weights file path (default: ema_bf16.safetensors in same directory as input file)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force overwrite if output file already exists"
    )

    args = parser.parse_args()

    # Determine input file path
    if args.input_path:
        input_file = Path(args.input_path)
    else:
        # Find ema.safetensors from checkpoint directory
        checkpoint_dir = Path(args.checkpoint_dir)
        if not checkpoint_dir.exists():
            print(f"Error: Checkpoint directory does not exist: {checkpoint_dir}")
            sys.exit(1)

        input_file = checkpoint_dir / "ema.safetensors"
        if not input_file.exists():
            print(f"Error: ema.safetensors file not found in checkpoint directory: {input_file}")
            sys.exit(1)

    # Determine output file path
    if args.output_path:
        output_file = Path(args.output_path)
    else:
        # Default output path: ema_bf16.safetensors in same directory as input file
        output_file = input_file.parent / "ema_bf16.safetensors"

    # Check input file
    if not input_file.exists():
        print(f"Error: Input file does not exist: {input_file}")
        sys.exit(1)

    # Check if output file already exists
    if output_file.exists() and not args.force:
        print(f"Error: Output file already exists: {output_file}")
        print("Use --force parameter to force overwrite")
        sys.exit(1)

    print("=" * 60)
    print("EMA Weight Conversion Tool - Float32 to BFloat16")
    print("=" * 60)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Input file size: {get_file_size_mb(input_file):.2f} MB")
    print("-" * 60)

    # Execute conversion
    try:
        # 1. Load EMA weights
        tensors = load_ema_weights(input_file)
        if tensors is None:
            sys.exit(1)

        # 2. Convert to bf16
        bf16_tensors = convert_to_bf16(tensors)

        # 3. Save bf16 weights
        success = save_bf16_weights(bf16_tensors, output_file)

        if success:
            print("-" * 60)
            print("✓ Conversion completed!")
            print(f"Output file size: {get_file_size_mb(output_file):.2f} MB")
            print(
                f"File size reduced: {get_file_size_mb(input_file) - get_file_size_mb(output_file):.2f} MB"
            )
            print("=" * 60)
            sys.exit(0)
        else:
            print("✗ Conversion failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nUser interrupted operation")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
