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
