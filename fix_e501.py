#!/usr/bin/env python3
"""Script to fix E501 (line too long) errors automatically."""

import re
import subprocess
from pathlib import Path

def get_e501_errors(file_path: Path) -> list[tuple[int, str]]:
    """Get all E501 errors for a file."""
    result = subprocess.run(
        ['python', '-m', 'ruff', 'check', '--select', 'E501', str(file_path)],
        capture_output=True,
        text=True
    )
    
    errors = []
    for line in result.stdout.split('\n'):
        if 'E501' in line:
            match = re.search(r':(\d+):\d+:', line)
            if match:
                line_num = int(match.group(1))
                errors.append(line_num)
    
    return errors

def fix_long_line(line: str, max_length: int = 100) -> str:
    """Fix a long line by breaking it appropriately."""
    if len(line) <= max_length:
        return line
    
    # If it's a string, try to break it
    if 'f"' in line or "f'" in line:
        # Find f-strings and break them
        if 'f"' in line:
            parts = line.split('f"', 1)
            if len(parts) == 2:
                prefix = parts[0] + 'f"'
                content = parts[1].rsplit('"', 1)[0]
                suffix = '"' + parts[1].rsplit('"', 1)[1] if '"' in parts[1] else ''
                
                # Try to break at logical points
                if len(prefix + content + suffix) > max_length:
                    # Break the f-string
                    indent = len(line) - len(line.lstrip())
                    spaces = ' ' * (indent + 4)
                    
                    # Find a good break point
                    if '\\n' in content:
                        # Already has newlines, just add line continuation
                        return prefix + content + suffix
                    else:
                        # Try to break at commas, spaces, etc.
                        if ', ' in content:
                            parts = content.split(', ', 1)
                            return f'{prefix}{parts[0]}, "\\n"\n{spaces}f"{parts[1]}{suffix}'
    
    # For other cases, try to break at operators or commas
    indent = len(line) - len(line.lstrip())
    spaces = ' ' * (indent + 4)
    
    # Try breaking at common patterns
    for pattern in [', ', ' + ', ' = ', ' += ', ' -= ']:
        if pattern in line and len(line) > max_length:
            parts = line.split(pattern, 1)
            if len(parts[0]) < max_length - 10:
                return f'{parts[0]}{pattern}\n{spaces}{parts[1]}'
    
    return line

def fix_file(file_path: Path) -> bool:
    """Fix E501 errors in a file."""
    errors = get_e501_errors(file_path)
    if not errors:
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    modified = False
    for line_num in sorted(set(errors), reverse=True):
        if line_num <= len(lines):
            original = lines[line_num - 1]
            fixed = fix_long_line(original.rstrip('\n'))
            if fixed != original.rstrip('\n'):
                lines[line_num - 1] = fixed + '\n'
                modified = True
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        return True
    return False

if __name__ == '__main__':
    # Get all files with E501 errors
    result = subprocess.run(
        ['python', '-m', 'ruff', 'check', '--select', 'E501', 
         'python/data_processor', 'python/tests'],
        capture_output=True,
        text=True
    )
    
    files_with_errors = set()
    for line in result.stdout.split('\n'):
        if 'E501' in line and 'python/' in line:
            match = re.search(r'(python/[^:]+):\d+:', line)
            if match:
                files_with_errors.add(Path(match.group(1)))
    
    for file_path in sorted(files_with_errors):
        print(f'Fixing {file_path}...')
        if fix_file(file_path):
            print(f'  Fixed {file_path}')

