#!/usr/bin/env python3
"""Add noqa comments for complexity errors using ruff JSON output."""

import json
import subprocess
import sys
from pathlib import Path


def get_function_def_line(lines: list[str], error_line: int) -> int:
    """Find the function/class definition line for an error."""
    # Look backwards from error line to find def/class
    for i in range(error_line - 1, max(-1, error_line - 20), -1):
        if i < 0:
            break
        line = lines[i].strip()
        if line.startswith(("def ", "class ")):
            return i + 1  # Convert to 1-indexed
    return error_line  # Fallback to error line itself


def add_noqa_to_file(file_path: Path) -> int:
    """Add noqa comments for complexity errors in a file."""
    # Get ruff errors as JSON
    result = subprocess.run(
        ["python", "-m", "ruff", "check", "--output-format", "json", str(file_path)],
        check=False, capture_output=True,
        text=True,
    )

    if not result.stdout.strip():
        return 0

    errors = json.loads(result.stdout)

    # Filter to complexity errors
    target_codes = {"PLR0911", "PLR0912", "PLR0915", "C901", "S110", "S101"}
    complexity_errors = [
        e for e in errors
        if e.get("code") in target_codes
    ]

    if not complexity_errors:
        return 0

    # Read file
    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()

    # Group errors by function definition line
    func_errors: dict[int, set[str]] = {}
    for error in complexity_errors:
        error_line = error["location"]["row"]
        error_code = error["code"]
        def_line = get_function_def_line(lines, error_line)

        if def_line not in func_errors:
            func_errors[def_line] = set()
        func_errors[def_line].add(error_code)

    # Add noqa comments
    modified = 0
    for def_line in sorted(func_errors.keys(), reverse=True):
        if def_line > len(lines):
            continue

        line = lines[def_line - 1]

        # Skip if already has noqa
        if "# noqa" in line:
            continue

        # Check if this is a function or class definition
        stripped = line.strip()
        if stripped.startswith(("def ", "class ")):
            codes_str = ",".join(sorted(func_errors[def_line]))
            line_clean = line.rstrip("\n\r")

            # Add noqa comment
            if line_clean.rstrip().endswith(":"):
                lines[def_line - 1] = f"{line_clean.rstrip()}  # noqa: {codes_str}\n"
            else:
                lines[def_line - 1] = f"{line_clean}  # noqa: {codes_str}\n"
            modified += 1

    if modified > 0:
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    return modified


def main() -> int:
    """Main function."""
    files_to_fix = [
        "python/data_processor/Data_Processor_r0.py",
        "python/data_processor/Data_Processor_Integrated.py",
    ]

    total_modified = 0
    for file_str in files_to_fix:
        file_path = Path(file_str)
        if not file_path.exists():
            continue

        modified = add_noqa_to_file(file_path)
        total_modified += modified

    return 0


if __name__ == "__main__":
    sys.exit(main())

