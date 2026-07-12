#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/sus/mlops"
API_COMPOSE="docker-compose.api.yml"
NETWORK_NAME="heart-disease-net"
HEALTH_URL="http://localhost:8000/health"
MAX_HEALTH_RETRIES=10
HEALTH_RETRY_DELAY=3

cd "$REPO_DIR"
git fetch origin
git checkout main
git pull origin main
CURRENT_SHA=$(git rev-parse --short HEAD)
echo "Deploying API at commit: $CURRENT_SHA"

if ! docker network inspect "$NETWORK_NAME" &> /dev/null; then
    docker network create "$NETWORK_NAME"
fi

docker compose -f "$API_COMPOSE" build
docker compose -f "$API_COMPOSE" up -d

for i in $(seq 1 "$MAX_HEALTH_RETRIES"); do
    if curl -sf "$HEALTH_URL" > /dev/null; then
        echo "API healthy (attempt $i)."
        exit 0
    fi
    sleep "$HEALTH_RETRY_DELAY"
done
echo "ERROR: API failed health check after $MAX_HEALTH_RETRIES attempts."
docker compose -f "$API_COMPOSE" logs --tail=50 api
exit 1
