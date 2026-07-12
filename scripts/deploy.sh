#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/sus/mlops}"
echo "Running manual fallback deployment at $REPO_DIR..."

cd "$REPO_DIR"

# Run API deployment
./scripts/deploy-api.sh

# Run Monitoring deployment
./scripts/deploy-monitoring.sh

echo "Fallback deployment complete!"
