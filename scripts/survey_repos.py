#!/usr/bin/env python3
"""
Survey external repositories for relevant modules and functions.
"""
import os
import re
import json

# List of external repositories to survey (absolute paths)
REPOS = [
    '../lorentz-violation-pipeline',
    '../lqg-anec-framework',
    '../unified-lqg',
    '../unified-lqg-qft',
    '../warp-bubble-exotic-matter-density',
    '../warp-bubble-qft'
]

OUTPUT_FILE = 'results/external_survey.json'

def survey_repos():
    survey = {}
    for repo in REPOS:
        repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), repo))
        survey[repo] = []
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                        for match in re.finditer(r'^def\s+(\w+)\s*\(', content, re.MULTILINE):
                            survey[repo].append({'file': os.path.relpath(file_path, repo_path), 'function': match.group(1)})
                    except Exception:
                        continue
    # Write output
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as outf:
        json.dump(survey, outf, indent=2)
    print(f"Survey results written to {OUTPUT_FILE}")

if __name__ == '__main__':
    survey_repos()
