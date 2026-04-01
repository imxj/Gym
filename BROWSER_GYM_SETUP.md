# Browser Gym CUA Environment — Setup & Test Guide

## Prerequisites

- Python 3.12+
- Docker (for SpotHub CRM simulator)
- Access to a LiteLLM-compatible inference endpoint with vision models
- ~4GB RAM for SpotHub + NeMo-Gym servers

## 1. Clone repo and checkout the branch

```bash
git clone https://github.com/imxj/Gym.git
cd Gym
git checkout feat/litellm-vision
```

## 2. Install NeMo-Gym

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install browser agent and resource server dependencies:

```bash
cd responses_api_agents/browser_agent && pip install -e . && cd ../..
cd responses_api_models/openai_model && pip install -e . && cd ../..
cd resources_servers/browser_gym && pip install -e . && cd ../..
```

> If the above fails, check each directory for a `pyproject.toml` or `setup.py` and install accordingly. The `ng_run` command handles venv creation per-server automatically if you omit `+skip_venv_if_present=true`.

## 3. Start SpotHub (CRM simulator)

Load and run the Docker image:

```bash
docker load -i /path/to/hubspot-gym.tar.gz
docker run -d --name spothub -p 3001:3000 hubspot-gym:latest
```

Verify it's running:

```bash
curl -s http://localhost:3001 | head -5
```

You should see HTML from the SpotHub Next.js app.

## 4. Configure environment

Create `env.yaml` in the repo root:

```yaml
# API key for LiteLLM-compatible endpoint
cua_openai_api_key: <your-api-key>
cua_openai_org: ""
cua_anthropic_api_key: placeholder
cua_gemini_api_key: placeholder

# Browser concurrency
max_concurrent_browsers: 16
browser_pool_size: 4

# Save screenshots + JSON for debugging
cua_debug_trajectories: true
```

## 5. Update model endpoints (if needed)

Edit `resources_servers/browser_gym/configs/browser_gym.yaml` if your LiteLLM endpoint differs from the default. Key fields per model server:

```yaml
browser_gym_openai_model:
  responses_api_models:
    openai_model:
      openai_base_url: https://your-litellm-endpoint
      openai_api_key: ${cua_openai_api_key}
      openai_model: openai/openai/gpt-5.4       # or your model route
```

All 4 model servers (openai, sonnet, opus, gemini) follow the same pattern — they all route through the `openai_model` server type via the LiteLLM proxy.

## 6. Start NeMo-Gym servers

```bash
source .venv/bin/activate
ng_run "+config_paths=[resources_servers/browser_gym/configs/browser_gym.yaml]" +skip_venv_if_present=true
```

Wait for `All 9 / 9 servers ready!` message. This starts:
- 1 resource server (Playwright browser pool)
- 4 model servers (GPT-5.4, Sonnet 4.6, Opus 4.6, Gemini 3.1)
- 4 agent servers (one per model, all using vision adapter)

## 7. Run a single task test

Create a test JSONL:

```bash
cat > /tmp/test_task.jsonl << 'EOF'
{"responses_create_params": {"input": [{"role": "user", "content": "Create a new company with the following details: Name: TechFlow Solutions, Domain: techflow-solutions.com, Industry: Computer Software, Number of employees: 150."}]}, "verifier_metadata": {"task_id": "CRM-COMPANY-CREATE-012", "gym_url": "http://localhost:3001", "start_url": "http://localhost:3001", "viewport": {"width": 1280, "height": 720}}}
EOF
```

Run rollout collection:

```bash
ng_collect_rollouts \
  +agent_name=browser_openai_agent \
  +input_jsonl_fpath=/tmp/test_task.jsonl \
  +output_jsonl_fpath=results/test_output.jsonl \
  +num_repeats=1 \
  +num_samples_in_parallel=1
```

Expected output: `"mean/reward": 1.0`

## 8. Run with different models

Replace `+agent_name=` with:
- `browser_openai_agent` — GPT-5.4
- `browser_anthropic_opus_agent` — Claude Opus 4.6
- `browser_anthropic_sonnet_agent` — Claude Sonnet 4.6
- `browser_gemini_agent` — Gemini 3.1 Pro

## 9. Run full benchmark (50 tasks)

First update `resources_servers/browser_gym/data/example.jsonl` to point gym_url to `http://localhost:3001`:

```bash
sed -i '' 's|https://lite.spothub.rlgym.turing.com|http://localhost:3001|g' \
  resources_servers/browser_gym/data/example.jsonl
```

Then run all tasks:

```bash
ng_collect_rollouts \
  +agent_name=browser_openai_agent \
  +input_jsonl_fpath=resources_servers/browser_gym/data/example.jsonl \
  +output_jsonl_fpath=results/full_benchmark.jsonl \
  +num_repeats=1 \
  +num_samples_in_parallel=4
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  Agent Server    │───▶│  Model Server    │───▶│  LiteLLM Endpoint   │
│  (Vision Adapter)│    │  (Normalizer)    │    │  (GPT/Claude/Gemini)│
│                  │    │                  │    │                     │
│  - JSON actions  │    │  - chat.completion│    │                     │
│  - Screenshot    │    │    normalization  │    │                     │
│    history mgmt  │    │  - reasoning fix  │    │                     │
└────────┬─────────┘    └──────────────────┘    └─────────────────────┘
         │
         ▼
┌─────────────────┐    ┌──────────────────┐
│ Resource Server  │───▶│  SpotHub Docker  │
│ (Playwright)     │    │  (CRM on :3001)  │
│                  │    │                  │
│ - Browser pool   │    │ - 50 CRM tasks   │
│ - Screenshots    │    │ - localStorage   │
│ - Action exec    │    │   verification   │
└─────────────────┘    └──────────────────┘
```

## Key changes from upstream PR #946

1. **`vision_adapter.py`** — New adapter that works with any vision LLM. Sends screenshots as images, prompts model for JSON actions. No `computer_use` tools needed.
2. **`openai_model/app.py`** — Response normalization for LiteLLM proxies that return `chat.completion` hybrid format. Also fixes `reasoning.effort="none"` for GPT-5.4.
3. **`browser_gym.yaml`** — All model servers route through `openai_model` type pointed at LiteLLM endpoint. All agents use `cua_adapter_type: vision`.

## Troubleshooting

- **Port 11000 in use**: `lsof -ti:11000 | xargs kill -9` then retry
- **Head server crashed**: `ray stop --force`, wait 2s, retry `ng_run`
- **0 steps / empty response**: Check model server logs — likely a response format validation error
- **Docker SpotHub not responding**: `docker restart spothub`
