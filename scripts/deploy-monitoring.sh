#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/sus/mlops"
MONITORING_COMPOSE="docker-compose.monitoring.yml"
NETWORK_NAME="heart-disease-net"
GRAFANA_URL="http://localhost:3000/api/health"

cd "$REPO_DIR"
git fetch origin
git checkout main
git pull origin main
CURRENT_SHA=$(git rev-parse --short HEAD)
echo "Deploying monitoring at commit: $CURRENT_SHA"

if ! docker network inspect "$NETWORK_NAME" &> /dev/null; then
    docker network create "$NETWORK_NAME"
fi

# up -d only — never recreate volumes, never `down -v` here
docker compose -f "$MONITORING_COMPOSE" up -d

if curl -sf "$GRAFANA_URL" > /dev/null; then
    echo "Grafana healthy."
else
    echo "WARNING: Grafana did not respond — check logs:"
    docker compose -f "$MONITORING_COMPOSE" logs --tail=50 grafana
fi
