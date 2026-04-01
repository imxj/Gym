#!/bin/bash
# Quick setup script for Browser Gym CUA environment
# Usage: ./setup_browser_gym.sh <api-key> [litellm-endpoint]
#
# Example:
#   ./setup_browser_gym.sh sk-xxx https://your-litellm-endpoint
#   ./setup_browser_gym.sh sk-xxx  # defaults to https://inference-api.nvidia.com

set -euo pipefail

API_KEY="${1:?Usage: $0 <api-key> [litellm-endpoint]}"
ENDPOINT="${2:-https://inference-api.nvidia.com}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Browser Gym Setup ==="
echo "Endpoint: $ENDPOINT"
echo "Repo dir: $SCRIPT_DIR"

# 1. Create env.yaml
cat > "$SCRIPT_DIR/env.yaml" << EOF
cua_openai_api_key: ${API_KEY}
cua_openai_org: ""
cua_anthropic_api_key: placeholder
cua_gemini_api_key: placeholder
max_concurrent_browsers: 16
browser_pool_size: 4
cua_debug_trajectories: true
EOF
echo "[1/5] env.yaml created"

# 2. Update endpoint in config if non-default
if [ "$ENDPOINT" != "https://inference-api.nvidia.com" ]; then
  sed -i.bak "s|https://inference-api.nvidia.com|${ENDPOINT}|g" \
    "$SCRIPT_DIR/resources_servers/browser_gym/configs/browser_gym.yaml"
  rm -f "$SCRIPT_DIR/resources_servers/browser_gym/configs/browser_gym.yaml.bak"
  echo "[2/5] Config updated with custom endpoint"
else
  echo "[2/5] Config OK (default endpoint)"
fi

# 3. Point example tasks to localhost SpotHub
sed -i.bak 's|https://lite.spothub.rlgym.turing.com|http://localhost:3001|g' \
  "$SCRIPT_DIR/resources_servers/browser_gym/data/example.jsonl"
rm -f "$SCRIPT_DIR/resources_servers/browser_gym/data/example.jsonl.bak"
echo "[3/5] Task data pointed to localhost:3001"

# 4. Start SpotHub Docker
if docker ps --format '{{.Names}}' | grep -q '^spothub$'; then
  echo "[4/5] SpotHub already running"
else
  if docker ps -a --format '{{.Names}}' | grep -q '^spothub$'; then
    docker start spothub
  else
    echo "Loading SpotHub Docker image..."
    echo "  If not loaded yet, run: docker load -i /path/to/hubspot-gym.tar.gz"
    docker run -d --name spothub -p 3001:3000 hubspot-gym:latest 2>/dev/null || {
      echo "ERROR: hubspot-gym image not found. Load it first:"
      echo "  docker load -i /path/to/hubspot-gym.tar.gz"
      exit 1
    }
  fi
  # Wait for SpotHub to be ready
  for i in $(seq 1 30); do
    if curl -s http://localhost:3001 > /dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  echo "[4/5] SpotHub started on :3001"
fi

# 5. Start NeMo-Gym servers
echo "[5/5] Starting NeMo-Gym servers..."
cd "$SCRIPT_DIR"
source .venv/bin/activate 2>/dev/null || {
  echo "Creating venv..."
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -e . -q
}

ray stop --force 2>/dev/null || true
sleep 2

echo ""
echo "Run the following to start servers:"
echo "  cd $SCRIPT_DIR && source .venv/bin/activate"
echo "  ng_run \"+config_paths=[resources_servers/browser_gym/configs/browser_gym.yaml]\" +skip_venv_if_present=true"
echo ""
echo "Then in another terminal, run a test:"
echo "  cd $SCRIPT_DIR && source .venv/bin/activate"
echo "  ng_collect_rollouts +agent_name=browser_openai_agent \\"
echo "    +input_jsonl_fpath=resources_servers/browser_gym/data/example.jsonl \\"
echo "    +output_jsonl_fpath=results/benchmark.jsonl \\"
echo "    +num_repeats=1 +num_samples_in_parallel=4"
echo ""
echo "=== Setup complete ==="
