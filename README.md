# Heart Disease Classifier API

This project exposes the Heart Disease prediction model through a FastAPI service and sets up monitoring/observability using Prometheus and Grafana.

## Running the Stack

The API and monitoring services have separate Compose lifecycles and communicate over an external Docker network.

### 1. One-time Setup
Create the shared external Docker network before starting either stack:
```bash
docker network create heart-disease-net
```

### 2. Start the API Stack
Build and run the FastAPI service:
```bash
docker compose -f docker-compose.api.yml up -d --build
```

### 3. Start the Monitoring Stack
Bring up Prometheus and Grafana services:
```bash
docker compose -f docker-compose.monitoring.yml up -d
```
Grafana will be accessible at [http://localhost:3000](http://localhost:3000) (using `admin`/`admin` credentials) and Prometheus at [http://localhost:9090](http://localhost:9090).

### 4. Restarting Services Independently
- Restart the Grafana dashboard alone (e.g. after a query change) without touching the API:
  ```bash
  docker compose -f docker-compose.monitoring.yml restart grafana
  ```
- Tear down the monitoring stack without losing metrics or dashboard changes:
  ```bash
  docker compose -f docker-compose.monitoring.yml down
  ```
  *(Metrics and settings will persist in named Docker volumes).*

- Tear down and completely wipe all monitoring data:
  ```bash
  docker compose -f docker-compose.monitoring.yml down -v
  ```
