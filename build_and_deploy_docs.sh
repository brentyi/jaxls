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

# Execute notebooks on the GPU when one is available (jaxls is much faster on
# GPU, so the rendered solver timings are representative); fall back to the
# CPU-only docs extra otherwise. Notebook execution is serial
# (nb_execution_parallel = 1) so concurrent notebooks don't contend for the GPU.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "GPU detected; building docs with the docs-gpu extra."
    DOCS_EXTRA=docs-gpu
else
    echo "No GPU detected; building docs with the CPU-only docs extra."
    DOCS_EXTRA=docs
fi

echo "Building documentation..."
make docs DOCS_EXTRA="$DOCS_EXTRA"

if [[ "$DRY_RUN" == true ]]; then
    echo "Dry run complete. Built docs are in docs/build/dirhtml/"
    exit 0
fi

echo "Deploying to gh-pages..."
uvx ghp-import -n -p -f docs/build/dirhtml

echo "Done! Documentation deployed to gh-pages branch."
