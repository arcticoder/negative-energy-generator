#!/usr/bin/env bash
# Merge 'wip' branches into 'main' across multiple repositories and install MPB

set -euo pipefail

# Install MPB via mamba in physics-suite environment
echo "Installing MPB..."
mamba install -n physics-suite -c conda-forge mpb -y

# List of repositories to update
repos=(
  "/home/echo_/Code/asciimath/enhanced-simulation-hardware-abstraction-framework"
  "/home/echo_/Code/asciimath/lqg-anec-framework"
  "/home/echo_/Code/asciimath/lqg-first-principles-gravitational-constant"
  "/home/echo_/Code/asciimath/energy"
  "/home/echo_/Code/asciimath/lqg-ftl-metric-engineering"
)

for repo in "${repos[@]}"; do
  echo "Updating repository: $repo"
  cd "$repo"
  git checkout main
  git merge wip -m "Merge wip into main"
  git branch -d wip
done

# Push the final repository
cd "/home/echo_/Code/asciimath/lqg-ftl-metric-engineering"
echo "Pushing merged main branch..."
git push origin main

echo "All repositories merged successfully."
