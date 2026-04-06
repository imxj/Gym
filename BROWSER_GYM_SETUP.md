# Browser Gym CUA Environment — Setup & Test Guide

## Prerequisites

- Python 3.12+
- Docker (for SpotHub CRM simulator)
- ~4GB RAM for SpotHub + NeMo-Gym servers

## Quick Start

```bash
# 1. Start SpotHub Docker
docker load -i hubspot-gym.tar.gz   # if not already loaded
docker run -d --name spothub -p 3001:3000 hubspot-gym:latest

# 2. Point task data to local SpotHub
sed -i 's|https://lite.spothub.rlgym.turing.com|http://localhost:3001|g' \
  resources_servers/browser_gym/data/example.jsonl

# 3. Install & start servers
source .venv/bin/activate
ng_run "+config_paths=[resources_servers/browser_gym/configs/browser_gym.yaml]" +skip_venv_if_present=true

# 4. Run a test (in another terminal)
source .venv/bin/activate
ng_collect_rollouts \
  +agent_name=browser_openai_agent \
  +input_jsonl_fpath=resources_servers/browser_gym/data/example.jsonl \
  +output_jsonl_fpath=results/benchmark.jsonl \
  +num_repeats=1 +num_samples_in_parallel=4
```

## Configuration

### env.yaml

```yaml
cua_openai_api_key: sk-QGYbzQt60MzP1h4O_8DCtg
cua_openai_org: ""
cua_anthropic_api_key: placeholder
cua_gemini_api_key: placeholder
max_concurrent_browsers: 16
browser_pool_size: 4
cua_debug_trajectories: true
```

### Models (via LiteLLM at https://inference-api.nvidia.com)

| Agent Name | Model Route | Adapter |
|---|---|---|
| `browser_openai_agent` | `openai/openai/gpt-5.4` | vision |
| `browser_anthropic_sonnet_agent` | `azure/anthropic/claude-sonnet-4-6` | vision |
| `browser_anthropic_opus_agent` | `aws/anthropic/bedrock-claude-opus-4-6` | vision |
| `browser_gemini_agent` | `gcp/google/gemini-3.1-pro-preview` | vision |

## Running Tests

### Single task

```bash
# Company creation (easy — reliably passes)
cat > /tmp/test.jsonl << 'EOF'
{"responses_create_params": {"input": [{"role": "user", "content": "Create a new company with the following details: Name: TechFlow Solutions, Domain: techflow-solutions.com, Industry: Computer Software, Number of employees: 150."}]}, "verifier_metadata": {"task_id": "CRM-COMPANY-CREATE-012", "gym_url": "http://localhost:3001", "start_url": "http://localhost:3001", "viewport": {"width": 1280, "height": 720}}}
EOF

ng_collect_rollouts +agent_name=browser_openai_agent \
  +input_jsonl_fpath=/tmp/test.jsonl \
  +output_jsonl_fpath=results/test.jsonl \
  +num_repeats=1 +num_samples_in_parallel=1
```

### Full benchmark (50 tasks)

```bash
ng_collect_rollouts +agent_name=browser_openai_agent \
  +input_jsonl_fpath=resources_servers/browser_gym/data/example.jsonl \
  +output_jsonl_fpath=results/full_benchmark_gpt54.jsonl \
  +num_repeats=1 +num_samples_in_parallel=4
```

## Architecture

```
Agent Server ──▶ Model Server ──▶ LiteLLM (inference-api.nvidia.com)
(VisionAdapter)  (Normalizer)     GPT-5.4 / Claude / Gemini
     │
     ▼
Resource Server ──▶ SpotHub Docker (:3001)
(Playwright)        50 CRM tasks, localStorage verification
```

## Troubleshooting

- **Port in use**: `lsof -ti:11000 | xargs kill -9; ray stop --force`
- **0 steps**: Model server response validation error — check Ray logs
- **SpotHub down**: `docker restart spothub`
