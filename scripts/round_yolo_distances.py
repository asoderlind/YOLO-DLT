#!/usr/bin/env python3
"""
Script to round the distance values (last column) in YOLO format label files to 2 decimal places.

Usage:
    python round_yolo_distances.py /path/to/labels/directory
"""

import argparse
import sys
from pathlib import Path

from tqdm import tqdm


def round_distance_in_line(line):
    """
    Round the distance value (last column) in a YOLO format line to 2 decimal places.

    Args:
        line (str): A line from the label file

    Returns:
        str: Line with rounded distance value
    """
    parts = line.strip().split()
    if len(parts) >= 6:  # Ensure we have all required columns including distance
        # Round the last value (distance) to 2 decimal places
        parts[-1] = f"{float(parts[-1]):.2f}"
        return " ".join(parts) + "\n"
    return line


def process_label_file(file_path):
    """
    Process a single label file and round distance values.

    Args:
        file_path (Path): Path to the label file
    """
    try:
        # Read the file
        with open(file_path, "r") as f:
            lines = f.readlines()

        # Process each line
        processed_lines = []
        for line in lines:
            if line.strip():  # Skip empty lines
                processed_lines.append(round_distance_in_line(line))
            else:
                processed_lines.append(line)

        # Write back to the file
        with open(file_path, "w") as f:
            f.writelines(processed_lines)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Round distance values in YOLO format label files to 2 decimal places."
    )
    parser.add_argument("directory", type=str, help="Path to directory containing YOLO label files (.txt)")

    args = parser.parse_args()

    # Convert to Path object and validate
    directory_path = Path(args.directory)

    if not directory_path.exists():
        print(f"Error: Directory '{directory_path}' does not exist.")
        sys.exit(1)

    if not directory_path.is_dir():
        print(f"Error: '{directory_path}' is not a directory.")
        sys.exit(1)

    # Find all .txt files in the directory
    txt_files = list(directory_path.glob("*.txt"))

    if not txt_files:
        print(f"No .txt files found in '{directory_path}'")
        sys.exit(0)

    print(f"Found {len(txt_files)} .txt files to process...")

    # Process each file with progress bar
    for file_path in tqdm(txt_files, desc="Processing files", unit="file"):
        process_label_file(file_path)

    print(f"Successfully processed {len(txt_files)} files!")


if __name__ == "__main__":
    main()
