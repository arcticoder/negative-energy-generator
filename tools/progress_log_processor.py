#!/usr/bin/env python3
"""
Progress Log Processor

Processes the progress_log.md file to:
1. Rotate progress blocks (newest -> progress -> oldest)
2. Update file listings, repo listings, test results, and import skip counts
"""

import os
import re
import subprocess
import sys
from pathlib import Path


def run_command(cmd, cwd=None):
    """Run a shell command and return its output."""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            cwd=cwd,
            check=False
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        print(f"Error running command '{cmd}': {e}")
        return "", str(e), 1


def find_block_boundaries(content, start_marker, end_marker):
    """Find the start and end positions of a block marked by specific comments."""
    start_match = re.search(re.escape(start_marker), content)
    end_match = re.search(re.escape(end_marker), content)
    
    if not start_match or not end_match:
        return None, None
    
    return start_match.end(), end_match.start()


def extract_block_content(content, start_marker, end_marker):
    """Extract content between start and end markers."""
    start_pos, end_pos = find_block_boundaries(content, start_marker, end_marker)
    if start_pos is None or end_pos is None:
        return None
    
    # Extract content between markers, strip leading/trailing whitespace
    block_content = content[start_pos:end_pos].strip()
    # Remove leading newline if present
    if block_content.startswith('\n'):
        block_content = block_content[1:]
    
    return block_content


def replace_block_content(content, start_marker, end_marker, new_content):
    """Replace content between markers with new content."""
    start_pos, end_pos = find_block_boundaries(content, start_marker, end_marker)
    if start_pos is None or end_pos is None:
        print(f"Warning: Could not find block markers {start_marker} / {end_marker}")
        return content
    
    # Replace content between markers
    before = content[:start_pos]
    after = content[end_pos:]
    
    # Ensure proper formatting with newlines
    if new_content and not before.endswith('\n'):
        before += '\n'
    if new_content and not after.startswith('\n'):
        new_content += '\n'
    
    return before + new_content + after


def process_progress_blocks(content):
    """Process the progress block rotations."""
    print("Processing progress block rotations...")
    
    # Step 1: Extract newest progress content
    newest_content = extract_block_content(
        content, 
        "## NEWEST-PROGRESS-BEGIN", 
        "## NEWEST-PROGRESS-END"
    )
    
    if newest_content is None:
        print("Warning: Could not find NEWEST-PROGRESS block")
        return content
    
    print(f"Extracted newest progress: {newest_content[:50]}...")
    
    # Step 2: Find the progress code fence right below NEWEST-PROGRESS-END
    newest_end_pos = content.find("## NEWEST-PROGRESS-END")
    if newest_end_pos == -1:
        print("Warning: Could not find NEWEST-PROGRESS-END marker")
        return content
    
    # Look for ```progress after the newest end marker
    progress_start = content.find("```progress", newest_end_pos)
    if progress_start == -1:
        print("Warning: Could not find progress code fence after NEWEST-PROGRESS block")
        return content
    
    progress_end = content.find("```", progress_start + 11)  # +11 to skip past "```progress"
    if progress_end == -1:
        print("Warning: Could not find end of progress code fence")
        return content
    
    # Extract current progress content (between ```progress and ```)
    current_progress_start = content.find("\n", progress_start) + 1
    current_progress_content = content[current_progress_start:progress_end].strip()
    
    print(f"Current progress content: {current_progress_content[:50]}...")
    
    # Step 3: Update progress code fence with newest content
    new_progress_block = f"```progress\n{newest_content}\n```"
    content = content[:progress_start] + new_progress_block + content[progress_end + 3:]
    
    # Step 4: Clear the newest progress block (leave empty with one blank line)
    content = replace_block_content(
        content,
        "## NEWEST-PROGRESS-BEGIN",
        "## NEWEST-PROGRESS-END", 
        "\n"
    )
    
    # Step 5: Replace oldest progress with current progress content
    content = replace_block_content(
        content,
        "## OLDEST-PROGRESS-BEGIN",
        "## OLDEST-PROGRESS-END",
        current_progress_content
    )
    
    print("Progress block rotations completed.")
    return content


def update_file_listings(content, repo_root):
    """Update the file listings section."""
    print("Updating file listings...")
    
    find_cmd = (
        'find . -path "./.venv" -prune -o -type f -regex '
        r"'.*\.\(ps1\|py\|sh\|ndjson\|json\|md\|yml\|toml\|h5\|ini\)$' "
        '-print | while read file; do stat -c \'%Y %n\' "$file"; done | '
        'sort -nr | while read timestamp file; do '
        'echo "$(date -d @$timestamp \'+%Y-%m-%d %H:%M:%S\') $file"; done | head -n 40'
    )
    
    stdout, stderr, returncode = run_command(find_cmd, cwd=repo_root)
    
    if returncode != 0:
        print(f"Warning: File listing command failed: {stderr}")
        return content
    
    content = replace_block_content(
        content,
        "# LATEST-FILES-LIST-BEGIN",
        "# LATEST-FILES-LIST-END",
        stdout
    )
    
    print("File listings updated.")
    return content


def update_repo_listings(content, repo_root):
    """Update the repository listings section."""
    print("Updating repo listings...")
    
    ls_cmd = "ls .. -lt | awk '{print $1, $2, $5, $6, $7, $8, $9}'"
    
    stdout, stderr, returncode = run_command(ls_cmd, cwd=repo_root)
    
    if returncode != 0:
        print(f"Warning: Repo listing command failed: {stderr}")
        return content
    
    content = replace_block_content(
        content,
        "# REPO-LIST-BEGIN", 
        "# REPO-LIST-END",
        stdout
    )
    
    print("Repo listings updated.")
    return content


def update_pytest_results(content, repo_root):
    """Update the pytest results section."""
    print("Updating pytest results...")
    
    # Activate virtual environment and run pytest
    pytest_cmd = 'source .venv/bin/activate && python -m pytest --maxfail=1'
    
    stdout, stderr, returncode = run_command(pytest_cmd, cwd=repo_root)
    
    # Combine stdout and stderr for complete test output
    test_output = stdout
    if stderr:
        test_output += f"\n{stderr}"
    
    if not test_output.strip():
        test_output = f"No test output (return code: {returncode})"
    
    content = replace_block_content(
        content,
        "# PYTEST-RESULTS-BEGIN",
        "# PYTEST-RESULTS-END", 
        test_output
    )
    
    print("Pytest results updated.")
    return content


def update_importerskip_count(content, asciimath_root):
    """Update the importerskip count section."""
    print("Updating importerskip count...")
    
    grep_cmd = 'grep -r "importerskip" --include="*.py" . | wc -l'
    
    stdout, stderr, returncode = run_command(grep_cmd, cwd=asciimath_root)
    
    if returncode != 0:
        print(f"Warning: Importerskip grep command failed: {stderr}")
        return content
    
    content = replace_block_content(
        content,
        "# IMPORTERSKIP-RESULTS-BEGIN",
        "# IMPORTERSKIP-RESULTS-END",
        stdout
    )
    
    print("Importerskip count updated.")
    return content


def main():
    """Main processing function."""
    # Determine paths
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    asciimath_root = repo_root.parent
    progress_log_path = repo_root / "docs" / "progress_log.md"
    
    print(f"Script directory: {script_dir}")
    print(f"Repository root: {repo_root}")
    print(f"Progress log path: {progress_log_path}")
    
    # Check if progress log exists
    if not progress_log_path.exists():
        print(f"Error: Progress log not found at {progress_log_path}")
        sys.exit(1)
    
    # Read the progress log
    try:
        with open(progress_log_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading progress log: {e}")
        sys.exit(1)
    
    print(f"Read {len(content)} characters from progress log.")
    
    # Process all updates
    try:
        # 1. Process progress block rotations
        content = process_progress_blocks(content)
        
        # 2. Update file listings
        content = update_file_listings(content, repo_root)
        
        # 3. Update repo listings  
        content = update_repo_listings(content, repo_root)
        
        # 4. Update pytest results
        content = update_pytest_results(content, repo_root)
        
        # 5. Update importerskip count
        content = update_importerskip_count(content, asciimath_root)
        
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)
    
    # Write the updated content back
    try:
        with open(progress_log_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully updated progress log: {progress_log_path}")
    except Exception as e:
        print(f"Error writing progress log: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()