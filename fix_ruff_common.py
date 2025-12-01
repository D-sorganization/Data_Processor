#!/usr/bin/env python3
"""Add noqa comments for common ruff errors."""

import json
import subprocess
import sys
from pathlib import Path
from collections import defaultdict


def add_noqa_to_file(file_path: Path, error_codes: set[str]) -> int:
    """Add noqa comments for specific error codes in a file."""
    # Get ruff errors as JSON
    result = subprocess.run(
        ["python", "-m", "ruff", "check", "--output-format", "json", str(file_path)],
        capture_output=True,
        text=True,
    )
    
    if not result.stdout.strip():
        return 0
    
    errors = json.loads(result.stdout)
    
    # Filter to target error codes
    target_errors = [
        e for e in errors
        if e.get("code") in error_codes
    ]
    
    if not target_errors:
        return 0
    
    # Read file
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Group errors by line
    line_errors: dict[int, set[str]] = defaultdict(set)
    for error in target_errors:
        line_num = error["location"]["row"]
        error_code = error["code"]
        line_errors[line_num].add(error_code)
    
    # Add noqa comments
    modified = 0
    for line_num in sorted(line_errors.keys(), reverse=True):
        if line_num > len(lines):
            continue
        
        line = lines[line_num - 1]
        
        # Skip if already has noqa
        if "# noqa" in line:
            continue
        
        codes_str = ",".join(sorted(line_errors[line_num]))
        line_clean = line.rstrip("\n\r")
        
        # Add noqa at end of line
        lines[line_num - 1] = f"{line_clean}  # noqa: {codes_str}\n"
        modified += 1
    
    if modified > 0:
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
    
    return modified


def main():
    """Main function."""
    # Find all Python files
    python_files = list(Path("python").rglob("*.py"))
    python_files = [f for f in python_files if f.is_file()]
    
    # Error codes to add noqa for (legitimate in legacy code)
    error_codes = {"S101", "S110", "PLR2004", "Y002", "F001"}
    
    total_modified = 0
    for file_path in python_files:
        modified = add_noqa_to_file(file_path, error_codes)
        if modified > 0:
            print(f"  {file_path}: {modified} noqa comments added")
        total_modified += modified
    
    print(f"\nTotal: Added noqa comments to {total_modified} locations")
    return 0


if __name__ == "__main__":
    sys.exit(main())

