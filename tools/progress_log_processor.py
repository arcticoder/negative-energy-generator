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
    
    # Step 5: Find the progress code fence directly above OLDEST-PROGRESS-BEGIN
    oldest_begin_pos = content.find("## OLDEST-PROGRESS-BEGIN")
    if oldest_begin_pos == -1:
        print("Warning: Could not find OLDEST-PROGRESS-BEGIN marker")
        return content
    
    # Look backwards for the closest ```progress before OLDEST-PROGRESS-BEGIN
    # Find all ```progress blocks before the oldest marker
    progress_blocks = []
    search_pos = 0
    while search_pos < oldest_begin_pos:
        progress_start_pos = content.find("```progress", search_pos)
        if progress_start_pos == -1 or progress_start_pos >= oldest_begin_pos:
            break
        progress_end_pos = content.find("```", progress_start_pos + 11)
        if progress_end_pos == -1:
            break
        progress_blocks.append((progress_start_pos, progress_end_pos))
        search_pos = progress_end_pos + 3
    
    if not progress_blocks:
        print("Warning: Could not find any progress code fence before OLDEST-PROGRESS block")
        return content
    
    # Get the last (closest) progress block before OLDEST-PROGRESS-BEGIN
    last_progress_start, last_progress_end = progress_blocks[-1]
    
    # Extract content from the progress block directly above OLDEST-PROGRESS
    progress_content_start = content.find("\n", last_progress_start) + 1
    progress_above_oldest = content[progress_content_start:last_progress_end].strip()
    
    print(f"Progress content above oldest: {progress_above_oldest[:50]}...")
    
    # Step 6: Replace oldest progress with content from progress block above it
    content = replace_block_content(
        content,
        "## OLDEST-PROGRESS-BEGIN",
        "## OLDEST-PROGRESS-END",
        progress_above_oldest
    )
    
    # Step 7: Delete the progress code fence that we just copied from
    content = content[:last_progress_start] + content[last_progress_end + 3:]
    
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
    
    # Use bash explicitly to handle source command
    pytest_cmd = '/bin/bash -c "source .venv/bin/activate && python -m pytest --maxfail=1"'
    
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


def update_importerskip_count(content, repo_root):
    """Update the importerskip count section - COMPLETELY REWRITTEN VERSION."""
    print("UPDATE_IMPORTERSKIP_COUNT: STARTING DEBUG VERSION")
    
    # Run a simple test first
    print(f"Working directory: {repo_root}")
    print(f"Directory exists: {os.path.exists(repo_root)}")
    
    # Test the basic grep command first
    basic_cmd = 'grep -r "importerskip" --include="*.py" . | wc -l'
    print(f"Running basic command: {basic_cmd}")
    basic_stdout, basic_stderr, basic_returncode = run_command(basic_cmd, cwd=repo_root)
    print(f"Basic result: stdout='{basic_stdout}', stderr='{basic_stderr}', returncode={basic_returncode}")
    
    # Test with exclude-dir
    exclude_cmd = 'grep -r "importerskip" --include="*.py" --exclude-dir=tools . | wc -l'
    print(f"Running exclude command: {exclude_cmd}")
    exclude_stdout, exclude_stderr, exclude_returncode = run_command(exclude_cmd, cwd=repo_root)
    print(f"Exclude result: stdout='{exclude_stdout}', stderr='{exclude_stderr}', returncode={exclude_returncode}")
    
    # Show what files are found
    files_cmd = 'grep -r "importerskip" --include="*.py" --exclude-dir=tools .'
    print(f"Running files command: {files_cmd}")
    files_stdout, files_stderr, files_returncode = run_command(files_cmd, cwd=repo_root)
    print(f"Files found: stdout='{files_stdout[:200]}...', stderr='{files_stderr}', returncode={files_returncode}")
    
    # Use the exclude result
    final_count = exclude_stdout
    print(f"Using final count: {final_count}")
    
    content = replace_block_content(
        content,
        "# IMPORTERSKIP-RESULTS-BEGIN",
        "# IMPORTERSKIP-RESULTS-END",
        final_count
    )
    
    print("UPDATE_IMPORTERSKIP_COUNT: COMPLETED")
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
        print("About to call update_importerskip_count...")
        content = update_importerskip_count(content, repo_root)
        print("Finished calling update_importerskip_count...")
        
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