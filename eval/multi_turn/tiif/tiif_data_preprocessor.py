# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd


def load_and_validate_tiif_data(input_dir):
    """
    Load and validate TIIF-Bench data from jsonl files.
    Returns validated data and statistics.
    """
    data = []
    stats = {
        'total_files': 0,
        'total_items': 0,
        'types': Counter(),
        'files_by_type': defaultdict(list),
        'items_per_file': {},
        'validation_errors': []
    }
    
    jsonl_files = list(Path(input_dir).glob("*.jsonl"))
    stats['total_files'] = len(jsonl_files)
    
    print(f"Found {len(jsonl_files)} jsonl files in {input_dir}")
    
    for jsonl_file in jsonl_files:
        file_items = []
        file_stats = {
            'file_name': jsonl_file.name,
            'items': 0,
            'errors': []
        }
        
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_idx, line in enumerate(f):
                    if line.strip():  # Skip empty lines
                        try:
                            item = json.loads(line)
                            
                            # Validate required fields
                            required_fields = ['type', 'short_description', 'long_description']
                            missing_fields = [field for field in required_fields if field not in item]
                            
                            if missing_fields:
                                error_msg = f"Line {line_idx}: Missing fields {missing_fields}"
                                file_stats['errors'].append(error_msg)
                                stats['validation_errors'].append(f"{jsonl_file.name}:{error_msg}")
                                continue
                            
                            # Add metadata for tracking
                            item['file_name'] = jsonl_file.name
                            item['line_idx'] = line_idx
                            
                            # Validate data types and content
                            if not isinstance(item['type'], str) or not item['type'].strip():
                                error_msg = f"Line {line_idx}: Invalid type field"
                                file_stats['errors'].append(error_msg)
                                continue
                                
                            if not isinstance(item['short_description'], str) or not item['short_description'].strip():
                                error_msg = f"Line {line_idx}: Invalid short_description field"
                                file_stats['errors'].append(error_msg)
                                continue
                                
                            if not isinstance(item['long_description'], str) or not item['long_description'].strip():
                                error_msg = f"Line {line_idx}: Invalid long_description field"
                                file_stats['errors'].append(error_msg)
                                continue
                            
                            # Update statistics
                            stats['types'][item['type']] += 1
                            stats['files_by_type'][item['type']].append(jsonl_file.name)
                            
                            file_items.append(item)
                            file_stats['items'] += 1
                            
                        except json.JSONDecodeError as e:
                            error_msg = f"Line {line_idx}: JSON decode error - {e}"
                            file_stats['errors'].append(error_msg)
                            stats['validation_errors'].append(f"{jsonl_file.name}:{error_msg}")
                            
        except Exception as e:
            error_msg = f"Error reading file {jsonl_file.name}: {e}"
            file_stats['errors'].append(error_msg)
            stats['validation_errors'].append(error_msg)
        
        data.extend(file_items)
        stats['items_per_file'][jsonl_file.name] = file_stats
        stats['total_items'] += file_stats['items']
        
        print(f"  {jsonl_file.name}: {file_stats['items']} items, {len(file_stats['errors'])} errors")
    
    return data, stats


def generate_statistics_report(stats, output_file=None):
    """Generate a detailed statistics report."""
    report = []
    report.append("TIIF-Bench Data Statistics Report")
    report.append("=" * 50)
    report.append(f"Total files processed: {stats['total_files']}")
    report.append(f"Total valid items: {stats['total_items']}")
    report.append(f"Total validation errors: {len(stats['validation_errors'])}")
    report.append("")
    
    # Type distribution
    report.append("Type Distribution:")
    report.append("-" * 20)
    for type_name, count in stats['types'].most_common():
        report.append(f"  {type_name}: {count} items")
    report.append("")
    
    # File statistics
    report.append("File Statistics:")
    report.append("-" * 20)
    for file_name, file_stats in stats['items_per_file'].items():
        report.append(f"  {file_name}: {file_stats['items']} items, {len(file_stats['errors'])} errors")
    report.append("")
    
    # Validation errors
    if stats['validation_errors']:
        report.append("Validation Errors:")
        report.append("-" * 20)
        for error in stats['validation_errors'][:20]:  # Show first 20 errors
            report.append(f"  {error}")
        if len(stats['validation_errors']) > 20:
            report.append(f"  ... and {len(stats['validation_errors']) - 20} more errors")
        report.append("")
    
    # Description length statistics
    report.append("Description Length Analysis:")
    report.append("-" * 30)
    # This would require the actual data to compute, so we'll add it later if needed
    
    report_text = "\n".join(report)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"Statistics report saved to {output_file}")
    
    return report_text


def create_sample_subset(data, output_file, sample_size=None, sample_ratio=None):
    """Create a sample subset of the data for testing."""
    if sample_size is None and sample_ratio is None:
        raise ValueError("Either sample_size or sample_ratio must be specified")
    
    if sample_ratio is not None:
        sample_size = int(len(data) * sample_ratio)
    
    if sample_size >= len(data):
        print(f"Sample size {sample_size} >= total data size {len(data)}, using all data")
        sample_data = data
    else:
        import random
        random.seed(42)  # For reproducibility
        sample_data = random.sample(data, sample_size)
    
    # Group by type to maintain distribution
    type_groups = defaultdict(list)
    for item in sample_data:
        type_groups[item['type']].append(item)
    
    # Save sample data maintaining the original jsonl format
    with open(output_file, 'w', encoding='utf-8') as f:
        for type_name, items in type_groups.items():
            for item in items:
                # Remove metadata fields before saving
                clean_item = {k: v for k, v in item.items() if k not in ['file_name', 'line_idx']}
                f.write(json.dumps(clean_item, ensure_ascii=False) + '\n')
    
    print(f"Sample subset with {len(sample_data)} items saved to {output_file}")
    
    # Print sample statistics
    print("Sample distribution:")
    for type_name, items in type_groups.items():
        print(f"  {type_name}: {len(items)} items")


def export_to_csv(data, output_file):
    """Export data to CSV format for analysis."""
    # Prepare data for CSV
    csv_data = []
    for item in data:
        csv_data.append({
            'type': item['type'],
            'file_name': item['file_name'],
            'line_idx': item['line_idx'],
            'short_description': item['short_description'],
            'long_description': item['long_description'],
            'short_length': len(item['short_description']),
            'long_length': len(item['long_description']),
            'length_ratio': len(item['long_description']) / len(item['short_description']) if item['short_description'] else 0
        })
    
    df = pd.DataFrame(csv_data)
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Data exported to CSV: {output_file}")
    
    # Print some basic statistics
    print("\nDescription Length Statistics:")
    print(f"Short descriptions - Mean: {df['short_length'].mean():.1f}, Std: {df['short_length'].std():.1f}")
    print(f"Long descriptions - Mean: {df['long_length'].mean():.1f}, Std: {df['long_length'].std():.1f}")
    print(f"Length ratio - Mean: {df['length_ratio'].mean():.1f}, Std: {df['length_ratio'].std():.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess and validate TIIF-Bench data for multi-turn generation."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing TIIF-Bench jsonl files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tiif_preprocessed",
        help="Directory to save preprocessed data and reports."
    )
    parser.add_argument(
        "--create_sample",
        action="store_true",
        help="Create a sample subset for testing."
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of items in the sample subset."
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=None,
        help="Ratio of items to include in the sample subset (0.0-1.0)."
    )
    parser.add_argument(
        "--export_csv",
        action="store_true",
        help="Export data to CSV format for analysis."
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load and validate data
    print("Loading and validating TIIF-Bench data...")
    data, stats = load_and_validate_tiif_data(args.input_dir)
    
    if not data:
        print("No valid data found. Exiting.")
        return
    
    # Generate statistics report
    report_file = output_path / "statistics_report.txt"
    report = generate_statistics_report(stats, report_file)
    print("\nStatistics Report:")
    print(report)
    
    # Create sample subset if requested
    if args.create_sample:
        sample_file = output_path / "sample_data.jsonl"
        create_sample_subset(data, sample_file, args.sample_size, args.sample_ratio)
    
    # Export to CSV if requested
    if args.export_csv:
        csv_file = output_path / "tiif_data_analysis.csv"
        export_to_csv(data, csv_file)
    
    print(f"\nPreprocessing completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
