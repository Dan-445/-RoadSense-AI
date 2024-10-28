import os
import sys
from pathlib import Path
import argparse

def add_prefix_to_filenames(directory, prefix, pattern_start='frame', pattern_end='.png'):
    """
    Adds a prefix to filenames in the specified directory that match the given pattern.

    Parameters:
    - directory (str): Path to the directory containing the image files.
    - prefix (str): The prefix to add to each filename.
    - pattern_start (str): The starting string of filenames to match.
    - pattern_end (str): The ending string of filenames to match (file extension).

    Example:
    If a file is named 'frame000.png' and prefix is '41',
    it will be renamed to '41frame000.png'.
    """
    path = Path(directory)
    if not path.is_dir():
        print(f"Error: The directory '{directory}' does not exist or is not a directory.")
        sys.exit(1)

    for file in path.iterdir():
        if file.is_file() and file.name.startswith(pattern_start) and file.name.endswith(pattern_end):
            new_name = f"{prefix}{file.name}"
            new_file = file.parent / new_name

            # Check if the new filename already exists to avoid overwriting
            if new_file.exists():
                print(f"Warning: '{new_name}' already exists. Skipping '{file.name}'.")
                continue

            try:
                file.rename(new_file)
                print(f"Renamed: '{file.name}' --> '{new_name}'")
            except Exception as e:
                print(f"Error renaming '{file.name}': {e}")

def main():
    """
    Main function to execute the renaming process.
    """
    parser = argparse.ArgumentParser(description="Add a prefix to image filenames in a directory.")
    parser.add_argument('--directory', type=str, required=True, help="Path to the target directory.")
    parser.add_argument('--prefix', type=str, required=True, help="Prefix to add to each filename.")
    parser.add_argument('--pattern_start', type=str, default='frame', help="Start pattern of filenames to match.")
    parser.add_argument('--pattern_end', type=str, default='.png', help="End pattern of filenames to match (file extension).")

    args = parser.parse_args()

    add_prefix_to_filenames(args.directory, args.prefix, args.pattern_start, args.pattern_end)

if __name__ == "__main__":
    main()
