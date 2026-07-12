#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/sus/mlops}"
echo "Setting up VM for MLOps at $REPO_DIR..."

# Update and install basic dependencies
sudo apt-get update -y
sudo apt-get install -y curl git apt-transport-https ca-certificates gnupg lsb-release

# Install Docker if not installed
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    rm get-docker.sh
fi

# Install Docker Compose if not installed
if ! docker compose version &> /dev/null; then
    echo "Installing Docker Compose..."
    sudo apt-get install -y docker-compose-plugin
fi

# Start and enable Docker
sudo systemctl enable --now docker

# Configure firewall rules (UFW)
echo "Configuring UFW..."
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 8000/tcp    # FastAPI API
sudo ufw allow 3000/tcp    # Grafana
sudo ufw allow 9090/tcp    # Prometheus
sudo ufw --force enable

# Clone repo if not exists
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning repository..."
    sudo mkdir -p "$(dirname "$REPO_DIR")"
    sudo chown -R "$USER:$USER" "$(dirname "$REPO_DIR")"
    git clone https://github.com/sabarn0/dev-branch.git "$REPO_DIR"
fi

echo "VM setup complete!"
