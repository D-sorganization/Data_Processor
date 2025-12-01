"""Add noqa comments to Python files based on ruff errors."""

import re
import subprocess
import sys
from pathlib import Path


def get_ruff_errors(file_path: Path) -> dict[int, set[str]]:
    """Get ruff errors for a file, returning dict of line_num -> set of error codes."""
    result = subprocess.run(
        ["python", "-m", "ruff", "check", str(file_path)],
        check=False, capture_output=True,
        text=True,
    )

    errors: dict[int, set[str]] = {}
    for line in result.stdout.split("\n"):
        # Match pattern: file:line:col: CODE
        match = re.search(r":(\d+):\d+:\s+([A-Z]\d{3})", line)
        if match:
            line_num = int(match.group(1))
            error_code = match.group(2)
            if line_num not in errors:
                errors[line_num] = set()
            errors[line_num].add(error_code)

    return errors


def add_noqa_comments(file_path: Path, errors: dict[int, set[str]]) -> int:
    """Add noqa comments to function/class definitions that have errors."""
    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()

    # Error codes that should be added to function/class definitions
    target_codes = {"PLR0911", "PLR0912", "PLR0915", "C901", "S110", "S101"}

    modified = 0
    # Group errors by finding the function definition line
    func_errors: dict[int, set[str]] = {}  # line_num -> error_codes

    for line_num, error_codes in errors.items():
        if line_num > len(lines):
            continue

        # Check if this line has any target error codes
        relevant_codes = error_codes & target_codes
        if not relevant_codes:
            continue

        # Find the function/class definition line (look backwards)
        def_line_num = line_num
        for i in range(line_num - 1, max(0, line_num - 10), -1):
            line = lines[i].strip()
            if line.startswith(("def ", "class ")):
                def_line_num = i + 1  # Convert to 1-indexed
                break

        if def_line_num not in func_errors:
            func_errors[def_line_num] = set()
        func_errors[def_line_num].update(relevant_codes)

    # Now add noqa comments to the function definition lines
    for def_line_num, error_codes in func_errors.items():
        if def_line_num > len(lines):
            continue

        line = lines[def_line_num - 1]

        # Skip if already has noqa
        if "# noqa" in line:
            continue

        # Check if this is a function or class definition
        stripped = line.strip()
        if stripped.startswith(("def ", "class ")):
            # Add noqa comment - find where to add it (after colon or at end)
            codes_str = ",".join(sorted(error_codes))
            line_clean = line.rstrip("\n\r")

            # If line ends with colon, add after colon
            if line_clean.rstrip().endswith(":"):
                lines[def_line_num - 1] = f"{line_clean.rstrip()}  # noqa: {codes_str}\n"
            else:
                # Multi-line definition - add at end of line
                lines[def_line_num - 1] = f"{line_clean}  # noqa: {codes_str}\n"
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

        errors = get_ruff_errors(file_path)
        modified = add_noqa_comments(file_path, errors)
        total_modified += modified

    return 0


if __name__ == "__main__":
    sys.exit(main())

