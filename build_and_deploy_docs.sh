#!/bin/bash
set -e

# Build and deploy documentation to gh-pages branch.
#
# Usage:
#   ./build_and_deploy_docs.sh           # Build and deploy
#   ./build_and_deploy_docs.sh --dry-run # Build only, don't deploy

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
fi

echo "Building documentation..."
make docs

if [[ "$DRY_RUN" == true ]]; then
    echo "Dry run complete. Built docs are in docs/build/dirhtml/"
    exit 0
fi

echo "Deploying to gh-pages..."
uvx ghp-import -n -p -f docs/build/dirhtml

echo "Done! Documentation deployed to gh-pages branch."
