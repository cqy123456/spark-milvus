#!/usr/bin/env python3
"""
Convert SIFT1M HDF5 dataset to Parquet format for Spark

This script converts the HDF5-format SIFT1M dataset into Parquet files
that can be easily read by Spark without needing HDF5 libraries on executors.

Usage:
    python scripts/convert_hdf5_to_parquet.py [--hdf5-path PATH] [--output-dir DIR]

Requirements:
    pip install h5py pyarrow pandas
"""

import argparse
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path


def convert_hdf5_to_parquet(hdf5_path: str, output_dir: str):
    """Convert SIFT1M HDF5 dataset to Parquet format"""

    hdf5_path = Path(hdf5_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Opening HDF5 file: {hdf5_path}")

    with h5py.File(hdf5_path, 'r') as f:
        # Convert training vectors
        print("\nConverting training vectors...")
        train_data = f['/train'][:]
        print(f"  Shape: {train_data.shape}")
        print(f"  Dtype: {train_data.dtype}")

        # Convert to list of lists (Spark Vector format)
        train_vectors = train_data.tolist()

        # Create DataFrame
        train_df = pd.DataFrame({
            'features': train_vectors
        })

        # Write to Parquet
        train_output = output_dir / "sift1m-train.parquet"
        train_df.to_parquet(train_output, engine='pyarrow', compression='snappy')
        print(f"  ✓ Saved {len(train_vectors)} training vectors to {train_output}")

        # Convert test vectors
        print("\nConverting test vectors...")
        test_data = f['/test'][:]
        print(f"  Shape: {test_data.shape}")

        test_vectors = test_data.tolist()
        test_df = pd.DataFrame({
            'features': test_vectors
        })

        test_output = output_dir / "sift1m-test.parquet"
        test_df.to_parquet(test_output, engine='pyarrow', compression='snappy')
        print(f"  ✓ Saved {len(test_vectors)} test vectors to {test_output}")

        # Convert ground truth neighbors
        print("\nConverting ground truth...")
        neighbors_data = f['/neighbors'][:]
        print(f"  Shape: {neighbors_data.shape}")

        neighbors_list = neighbors_data.tolist()
        neighbors_df = pd.DataFrame({
            'neighbors': neighbors_list
        })

        neighbors_output = output_dir / "sift1m-groundtruth.parquet"
        neighbors_df.to_parquet(neighbors_output, engine='pyarrow', compression='snappy')
        print(f"  ✓ Saved {len(neighbors_list)} ground truth entries to {neighbors_output}")

    print("\n" + "=" * 70)
    print("Conversion completed successfully!")
    print("=" * 70)
    print(f"\nOutput files in: {output_dir.absolute()}")
    print(f"  - {train_output.name}")
    print(f"  - {test_output.name}")
    print(f"  - {neighbors_output.name}")
    print("\nYou can now use these Parquet files with Spark without HDF5 dependencies!")


def main():
    parser = argparse.ArgumentParser(
        description='Convert SIFT1M HDF5 dataset to Parquet format'
    )
    parser.add_argument(
        '--hdf5-path',
        type=str,
        default='./data/sift-128-euclidean.hdf5',
        help='Path to input HDF5 file (default: ./data/sift-128-euclidean.hdf5)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/parquet',
        help='Output directory for Parquet files (default: ./data/parquet)'
    )

    args = parser.parse_args()

    # Check if input file exists
    if not Path(args.hdf5_path).exists():
        print(f"ERROR: HDF5 file not found: {args.hdf5_path}")
        print("\nPlease download the dataset:")
        print("  wget http://ann-benchmarks.com/sift-128-euclidean.hdf5 -O data/sift-128-euclidean.hdf5")
        return 1

    convert_hdf5_to_parquet(args.hdf5_path, args.output_dir)
    return 0


if __name__ == '__main__':
    exit(main())
