# WebArena Resources Server

Classic [WebArena](https://webarena.dev) benchmark as a NeMo-Gym environment. Models browse
live self-hosted WebArena sites (shopping, shopping_admin/Magento, reddit/Postmill,
gitlab, wikipedia, map) through the shared `browser_gym` CUA machinery, and rewards come
from the classic WebArena evaluators. Ported from the internal standalone harness
(`osworld_internal/webarena/common/classic_evaluation.py`) so rewards match it.

## Architecture

This server **extends `resources_servers/browser_gym`** (browser pool, `/step`, `/close`,
`/dump_local_storage` are inherited) and pairs with the existing
`responses_api_agents/browser_agent` — no agent changes needed:

```
row (verifier_metadata: start_url, eval) -> browser_agent /run
    -> webarena /seed_session   substitutes __SHOPPING__-style placeholders,
                                logs into all configured sites (cookie-cached)
    -> webarena /step ...       inherited CUA actions (click/type/scroll/goto/tabs/...)
    -> webarena /verify         classic WebArena evaluation -> reward 0.0 | 1.0
```

`verify()` runs after the episode's browser session is closed, so it is self-sufficient:

| eval_type | how it is evaluated |
|---|---|
| `string_match` | `exact_match` / `must_include` locally; misses fall back to the LLM judge (`fuzzy_match`, `exact_match_fallback`, `must_include_fallback`, `ua_match` for `N/A` tasks). Reward logic ported 1:1 from the harness. |
| `url_match` | "GOLD in PRED" rule against the **trajectory's visited URLs** (most recent first) — the equivalent of checking all open tabs in the standalone harness. |
| `program_html` | Opens a **fresh authenticated browser session** via the pool, navigates to each target (`url: "last"` uses the trajectory URLs), evaluates `document.*` locators with `page.evaluate`, resolves `func:` site-API helpers (`site_api_helpers.py`: Magento REST orders/reviews/sales-rules, `reddit_get_post_url`, `gitlab_get_project_memeber_role`), and scores `required_contents` (`exact_match` / `must_include`, `|OR|` alternatives). |

The reward is the **product** over all `eval_types` (all must pass).

### Reward reliability: `mask_sample` + `termination_reason`

`verify()` returns a `WebArenaVerifyResponse` that extends the CUA verify
response with the unreliable-reward contract (same shape the OSWorld
integration uses): **`mask_sample=True` means "do not train on this reward"** —
the episode or its verification failed for an *infrastructure* reason, so the
0 says nothing about the policy. This formalizes the standalone harness's
"status=error, evaluation skipped" behavior.

| termination_reason | set by | masked by default |
|---|---|---|
| `browser_stuck`, `adapter_error`, `adapter_init_error`, `run_timeout` | browser agent loop | yes |
| `no_tool_calls`, `unparseable_action` | adapter parse-retry exhaustion | yes |
| `empty_trajectory` | verify (no steps, no answer) | yes |
| `judge_unavailable`, `judge_call_failed` | verify (fuzzy_match judge) | yes |
| `program_html_infra_error`, `site_api_error` | verify (site nav / Magento API) | yes |
| `verification_error` | verify (unexpected exception) | yes |
| `max_steps` | browser agent loop | **no** — running out of steps is a genuine failure (harness "fail") |

The set is configurable via `masked_termination_reasons`. Scoring is skipped
entirely for episodes arriving with a masked agent-side reason; a content
mismatch (wrong answer, wrong URL, wrong DOM state) never masks.

### Parse retries (harness parity)

Both adapters call the model up to `cua_parse_retries` (default 3) times per
step — i.e. up to 2 blind resamples after the first attempt, matching the
internal harness's 3-attempt loop — when the output carries no parseable
action, before ending the episode with a masked `no_tool_calls` /
`unparseable_action` reason. Failed attempts are never persisted to history,
so the trajectory and RL token stream only ever contain accepted turns.
`cua_parse_error_feedback: true` switches retries from blind resamples to
OSWorld-PR-style transient corrective messages (off by default to preserve
harness score parity); because the corrective messages are transient, a turn
recovered via feedback carries **no token IDs** — its prompt cannot be aligned
with the persisted history, and misaligned log-probs are worse than none.

Masking is **causal**: an infra flag raised during scoring only masks the
reward if it could have determined the outcome. A full-score episode stays
unmasked despite hiccups, and an episode with a clean zero (e.g. a genuine
url_match miss) stays unmasked even if an unrelated judge call failed.

`func:` expressions are parsed with `ast` (single call to an allow-listed helper,
literal args plus `__page__` / `__last_url__` placeholders) — no `eval`. Magento
API calls authenticate with the `shopping_admin` credentials and cache the admin
token per site; HTTP goes through NeMo-Gym's global aiohttp client.

### Known gaps vs the standalone harness (v1)

- No eval-collision snapshotting: concurrent **state-changing** rollouts against the same
  shared sites can pollute each other's evaluation. Start with `non_state_change` tasks
  (the converter tags each row) or run state-changing tasks with low concurrency and an
  out-of-band site reset between rounds.
- WebArena sites must be reachable and reset out-of-band (the standalone harness's
  `reset_webarena_env.sh` reset service); this server does not reset sites.

## Agent / policy interface

Two agent instances are configured; they differ only in the `browser_agent` adapter:

| agent | adapter | protocol | use for |
|---|---|---|---|
| `webarena_nemotron_agent` | `nemotron_toolcall` | **native tool calls**, 5 harness tools, normalized [0,1] coordinates, `terminate(status, answer)` | Nemotron/holotron checkpoints trained on internal browser trajectories; RL rollouts |
| `webarena_vision_agent` | `vision` | one JSON action object in plain text, pixel coordinates | frontier API models (GPT/Claude/Gemini) that were never trained on our tool schema |

`adapters/nemotron_toolcall_adapter.py` replicates the internal harness policy
interface (`osworld_internal webarena/nvidia/nemotron_toolcall_agent.py`): system
prompt and the five tool definitions are verbatim, `computer` carries a sequence
of actions, coordinates stay normalized until they are mapped onto pixel
`BrowserAction`s, and the terminal `terminate` answer becomes `final_message` —
which is exactly what `string_match` scores. History compaction is ported too:
only the last `cua_max_image_history` screenshots stay as images, and whole
oldest turns are dropped (with a redaction notice) to fit a text budget derived
from `cua_max_model_len`.

**Serving requirement:** this adapter needs vLLM started with
`--enable-auto-tool-choice` and the model's `--tool-call-parser`. Without them
the model's tool calls never reach the `tool_calls` field, every step parses as
"no tool call", and rewards read as 0 with no error.

Irreducible differences from the harness (worth weighing before comparing
numbers to standalone eval runs): observations are **headless Playwright
viewport** screenshots rather than full 1920x1080 X-display grabs (no browser
chrome), actions execute via Playwright rather than pyautogui, and there is no
Cloudflare/captcha handler.

## Configuration

All wiring is YAML (no env vars). `configs/webarena.yaml` expects these keys in the
Gym-root `env.yaml`:

```yaml
# WebArena sites (point at your self-hosted instance)
wa_shopping: http://<host>:7770
wa_shopping_admin: http://<host>:7780/admin
wa_reddit: http://<host>:9999
wa_gitlab: http://<host>:8023
wa_wikipedia: http://<host>:8888
wa_map: http://<host>:3000

# LLM judge for fuzzy_match (harness default: gpt-4.1 via inference-api)
webarena_judge_base_url: https://inference-api.nvidia.com
webarena_judge_api_key: <key>
webarena_judge_model_name: us/azure/openai/gpt-4.1

# Policy model
policy_base_url: http://localhost:8000/v1
policy_api_key: dummy
policy_model_name: <model>

webarena_max_concurrent_browsers: 16
webarena_browser_pool_size: 4
```

Site credentials default to the canonical WebArena demo accounts
(`schemas.DEFAULT_SITE_CREDENTIALS`) and can be overridden via `site_credentials`.
Omitting `judge_model_server` disables LLM fallbacks — judge-dependent checks then score
0.0 conservatively (fine for smoke tests on exact-match tasks, wrong for fuzzy tasks).

## Data schema

One task per JSONL line (converted from `osworld_internal/webarena/benchmarks/webarena.jsonl`
with `webarena/conversion/convert_webarena_to_nemo_gym.py` in that repo):

```json
{
  "responses_create_params": {"input": [{"role": "user", "content": "<intent>"}]},
  "verifier_metadata": {
    "task_id": "webarena-0",
    "sites": ["shopping_admin"],
    "start_url": "__SHOPPING_ADMIN__",
    "intent": "<intent>",
    "task_type": "non_state_change",
    "eval": {"eval_types": ["string_match"], "reference_answers": {"exact_match": "..."}}
  }
}
```

`start_url` and any URLs inside `eval` keep their `__SITE__` placeholders; this server
substitutes them from `site_urls` at seed/verify time, so the same data works against any
site deployment.

The full converted benchmark lives at `data/webarena_validation.jsonl` (812 tasks: 336
string_match / 197 url_match / 410 program_html; 338 tagged `non_state_change`) — it is
gitignored, regenerate it with the converter. For concurrent collection, filter to
`verifier_metadata.task_type == "non_state_change"` first (see gaps below).

## Run

```bash
ng_run "+config_paths=[resources_servers/webarena/configs/webarena.yaml]"

ng_collect_rollouts +agent_name=webarena_vision_agent \
  +input_jsonl_fpath=resources_servers/webarena/data/example.jsonl \
  +output_jsonl_fpath=results/webarena_rollouts.jsonl \
  +num_repeats=1

ng_reward_profile +input_jsonl_fpath=resources_servers/webarena/data/example.jsonl \
  +rollouts_jsonl_fpath=results/webarena_rollouts.jsonl \
  +output_jsonl_fpath=results/webarena_profiled.jsonl +pass_threshold=1.0
```

## Tests

Tier 0/1 only — no browser, no model, no sites:

```bash
cd resources_servers/webarena && pytest -x
```

## Smoke testing against live sites (no model, no GPU)

`tools/smoke_e2e.py` stages the expensive parts so failures isolate cheaply.
Run each from the Gym root; every stage writes a `*_smoke_stats.json`.

```bash
# 1. Playwright + browser pool work here at all (no network)
python resources_servers/webarena/tools/smoke_e2e.py browser

# 2. Scripted site logins — inspect the screenshots, don't trust the exit code
python resources_servers/webarena/tools/smoke_e2e.py login --sites shopping_admin gitlab reddit shopping

# 3. Evaluator vs live sites: score a synthetic *gold* trajectory per task
python resources_servers/webarena/tools/smoke_e2e.py verify-replay \
    --data resources_servers/webarena/data/webarena_validation.jsonl --limit 60
```

Stage 3 classifies every task so a single ratio can't hide a bug:

| outcome | meaning |
|---|---|
| `pass` | gold trajectory scored 1.0 — expected for exact_match / must_include / url_match |
| `needs_judge` | fuzzy_match task; unscorable without a judge model server wired up |
| `program_html_live_state` | check reads site state a gold trajectory never produced |
| `UNEXPECTED` | **a real bug in the evaluator port — must be 0** |

Reference run (2026-08-01, 60 tasks): `pass=41`, `needs_judge=19`, `UNEXPECTED=0`;
logins 4/4; browser OK.

### Site URLs drift

The addresses in the standalone harness's `webarena/nvidia/export_vars.sh`
(`10.131.133.31:*`) were unroutable as of 2026-08-01; the sites answered on the
EC2 host referenced by `reset_webarena_env.sh`. **Confirm the current host with
whoever owns the deployment** and set it once in `env.yaml` (`wa_*` keys) rather
than hard-coding it anywhere — this repo reads the values only from config.

Full rollouts (`ng_run` + `ng_collect_rollouts`) additionally need a served
policy model, and `ng_run` starts each server from its own per-server `.venv`;
run it in the environment those venvs were built for.
