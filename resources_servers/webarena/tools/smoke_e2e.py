#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Staged smoke tests for the WebArena Gym env that need NO model / no GPU.

Each stage fails loudly and cheaply before the next one is worth running.

    browser        Playwright + browser pool work here at all (no network).
    login          Seed a session against the real WebArena sites and confirm
                   the scripted logins produced an authenticated page.
    verify-replay  Feed a *synthetic gold trajectory* through verify() for each
                   task in a data file and check the reward is 1.0. This
                   exercises string_match / url_match / program_html (incl. the
                   func: site-API helpers) against the live sites without a
                   policy model.

Run from the Gym root with the site URLs configured in env.yaml, e.g.:

    uv run --no-project python resources_servers/webarena/tools/smoke_e2e.py browser
    uv run --no-project python resources_servers/webarena/tools/smoke_e2e.py login --sites shopping_admin gitlab
    uv run --no-project python resources_servers/webarena/tools/smoke_e2e.py verify-replay \
        --data resources_servers/webarena/data/example.jsonl

A <stage>_smoke_stats.json is written next to --out-dir with the command,
timestamp, and per-item results.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import datetime
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock


GYM_ROOT = Path(__file__).resolve().parents[3]
if str(GYM_ROOT) not in sys.path:
    sys.path.insert(0, str(GYM_ROOT))

from nemo_gym.server_utils import ServerClient  # noqa: E402
from resources_servers.browser_gym.schemas import (  # noqa: E402
    BrowserAction,
    CUANeMoGymResponse,
    CUAStep,
    CUATrajectory,
    CUAVerifyRequest,
)
from resources_servers.webarena.app import WebArenaResourcesServer  # noqa: E402
from resources_servers.webarena.evaluators import substitute_site_placeholders  # noqa: E402
from resources_servers.webarena.schemas import WebArenaResourcesServerConfig  # noqa: E402


def load_site_urls(env_yaml: Path) -> Dict[str, str]:
    """Read wa_* keys out of the Gym-root env.yaml (no hydra needed)."""
    mapping = {
        "wa_shopping": "shopping",
        "wa_shopping_admin": "shopping_admin",
        "wa_reddit": "reddit",
        "wa_gitlab": "gitlab",
        "wa_wikipedia": "wikipedia",
        "wa_map": "map",
        "wa_classifieds": "classifieds",
    }
    site_urls: Dict[str, str] = {}
    if not env_yaml.exists():
        return site_urls
    for line in env_yaml.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        site = mapping.get(key.strip())
        value = value.strip().strip('"').strip("'")
        if site and value:
            site_urls[site] = value
    return site_urls


def make_server(site_urls: Dict[str, str], **overrides) -> WebArenaResourcesServer:
    config = WebArenaResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="webarena_smoke",
        site_urls=site_urls,
        **overrides,
    )
    # verify() only needs the judge via server_client; the stages here use
    # tasks whose scoring is local, so a mock is enough.
    return WebArenaResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def write_stats(out_dir: Path, stage: str, payload: Dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "command": " ".join(sys.argv),
        "timestamp": datetime.datetime.now().isoformat(),
        "stage": stage,
        **payload,
    }
    path = out_dir / f"{stage}_smoke_stats.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return path


########################################
# Stage: browser
########################################


async def stage_browser(args) -> int:
    """Pool + Playwright sanity: seed a local page, click, screenshot, close."""
    server = make_server({}, login_sites_on_seed=False)
    page_html = "<html><body style='background:#eef'><h1 id='t'>gym browser smoke</h1></body></html>"
    data_url = "data:text/html;base64," + base64.b64encode(page_html.encode()).decode()

    env_id = "smoke-browser"
    result: Dict[str, Any] = {"ok": False}
    try:
        screenshot = await server.browser_pool.create_session(env_id=env_id, start_url=data_url)
        result["initial_screenshot_bytes"] = len(base64.b64decode(screenshot))

        shot, url, error = await server.browser_pool.execute_action(
            env_id, BrowserAction(action_type="click", coordinate=[100, 100], button="left")
        )
        result["after_click_screenshot_bytes"] = len(base64.b64decode(shot)) if shot else 0
        result["current_url_prefix"] = str(url)[:32]
        result["action_error"] = error
        result["ok"] = result["initial_screenshot_bytes"] > 1000 and not error
    finally:
        await server.browser_pool.close_session(env_id)
        await server.browser_pool.shutdown()

    if args.out_dir:
        write_stats(Path(args.out_dir), "browser", result)
    print("BROWSER SMOKE:", "OK" if result["ok"] else "FAILED")
    return 0 if result["ok"] else 1


########################################
# Stage: login
########################################


async def stage_login(args) -> int:
    """Seed a session per site and save a screenshot to eyeball the login."""
    site_urls = load_site_urls(Path(args.env_yaml))
    sites = args.sites or sorted(site_urls)
    missing = [s for s in sites if s not in site_urls]
    if missing:
        print(f"ERROR: no URL configured in {args.env_yaml} for: {missing}", file=sys.stderr)
        return 1

    server = make_server(site_urls)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []

    try:
        for site in sites:
            entry: Dict[str, Any] = {"site": site, "url": site_urls[site]}
            env_id = f"smoke-login-{site}"
            try:
                from resources_servers.webarena.schemas import WebArenaSeedSessionRequest

                response = await server.seed_session(WebArenaSeedSessionRequest(start_url=site_urls[site]))
                png = base64.b64decode(response.screenshot)
                path = out_dir / f"login_{site}.png"
                path.write_bytes(png)
                session = server.browser_pool.get_session(response.env_id)
                entry.update(
                    ok=len(png) > 1000,
                    screenshot=str(path),
                    screenshot_bytes=len(png),
                    final_url=session.page.url,
                    cookies_cached=site in server._auth_cookies,
                )
                await server.browser_pool.close_session(response.env_id)
            except Exception as e:  # noqa: BLE001 - smoke test reports, never raises
                entry.update(ok=False, error=f"{type(e).__name__}: {e}")
                await server.browser_pool.close_session(env_id)
            print(f"  {site}: {'OK' if entry.get('ok') else 'FAILED'} {entry.get('error', '')}")
            results.append(entry)
    finally:
        await server.browser_pool.shutdown()

    payload = {"sites": results, "num_ok": sum(1 for r in results if r.get("ok"))}
    write_stats(out_dir, "login", payload)
    print("\nOpen the screenshots and confirm each page shows a LOGGED-IN state.")
    return 0 if payload["num_ok"] == len(results) else 1


########################################
# Stage: verify-replay
########################################


def gold_trajectory(row: Dict[str, Any], site_urls: Dict[str, str]) -> CUANeMoGymResponse:
    """Synthesize the trajectory a perfect agent would have produced.

    - final_message = the reference answer (for string_match)
    - visited URLs   = the reference URL (for url_match)

    program_html is unaffected: it reads live site state, so those tasks only
    pass here if the site already satisfies the check.
    """
    vm = row["verifier_metadata"]
    eval_cfg = vm.get("eval") or {}
    refs = eval_cfg.get("reference_answers") or {}

    answer = ""
    if "exact_match" in refs:
        value = refs["exact_match"]
        answer = str(value[0] if isinstance(value, list) else value)
    elif "must_include" in refs:
        parts = [str(v[0] if isinstance(v, list) else v) for v in refs["must_include"]]
        answer = " ".join(parts)
    elif "fuzzy_match" in refs:
        value = refs["fuzzy_match"]
        answer = str(value if isinstance(value, str) else value[0])

    urls: List[str] = []
    reference_url = eval_cfg.get("reference_url")
    if reference_url:
        first = str(reference_url).split(" |OR| ")[0]
        urls.append(substitute_site_placeholders(first, site_urls))
    start_url = substitute_site_placeholders(vm.get("start_url", ""), site_urls)
    if start_url:
        urls.insert(0, start_url)

    steps = [
        CUAStep(
            action=BrowserAction(action_type="goto", url=url),
            screenshot_after="",
            current_url=url,
        )
        for url in urls
    ]
    return CUANeMoGymResponse(
        id="smoke_gold",
        created_at=0,
        model="gold",
        object="response",
        output=[],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        env_id="smoke",
        trajectory=CUATrajectory(steps=steps, task_prompt=vm.get("intent", ""), final_message=answer),
        local_storage_dump=None,
    )


async def stage_verify_replay(args) -> int:
    site_urls = load_site_urls(Path(args.env_yaml))
    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]

    server = make_server(site_urls)
    results: List[Dict[str, Any]] = []
    try:
        for row in rows:
            vm = row["verifier_metadata"]
            eval_types = (vm.get("eval") or {}).get("eval_types", [])
            request = CUAVerifyRequest(
                responses_create_params=row["responses_create_params"],
                response=gold_trajectory(row, site_urls),
                verifier_metadata=vm,
            )
            verify_response = await server.verify(request)
            entry = {
                "task_id": vm.get("task_id"),
                "eval_types": eval_types,
                "reward": verify_response.reward,
                "messages": (verify_response.verification_result or {}).get("messages", [])[:4],
            }
            flag = "OK  " if verify_response.reward == 1.0 else "MISS"
            print(f"  {flag} {entry['task_id']:<16} reward={entry['reward']} {eval_types}")
            results.append(entry)
    finally:
        await server.browser_pool.shutdown()

    def miss_reason(entry: Dict[str, Any]) -> str:
        if entry["reward"] == 1.0:
            return "pass"
        if any("judge unavailable" in m for m in entry["messages"]):
            return "needs_judge"  # fuzzy_match task; smoke run has no judge wired
        if "program_html" in entry["eval_types"]:
            return "program_html_live_state"  # gold trajectory never changed the site
        return "UNEXPECTED"

    for entry in results:
        entry["outcome"] = miss_reason(entry)

    counts: Dict[str, int] = {}
    for entry in results:
        counts[entry["outcome"]] = counts.get(entry["outcome"], 0) + 1

    payload = {
        "num_tasks": len(results),
        "num_reward_1": counts.get("pass", 0),
        "outcome_counts": counts,
        "results": results,
        "note": (
            "pass = gold trajectory scored 1.0 (expected for exact_match/must_include/url_match). "
            "needs_judge = fuzzy_match task, unscorable without a judge model server. "
            "program_html_live_state = check reads site state a gold trajectory never produced. "
            "UNEXPECTED = a real bug in the evaluator port; should be 0."
        ),
    }
    write_stats(Path(args.out_dir), "verify_replay", payload)
    print(f"\nOutcomes: {counts}")
    if counts.get("UNEXPECTED"):
        print(f"*** {counts['UNEXPECTED']} UNEXPECTED miss(es) — evaluator bug, investigate ***")
    return 1 if counts.get("UNEXPECTED") else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-yaml", default="env.yaml")
    parser.add_argument("--out-dir", default="results/webarena_smoke")
    sub = parser.add_subparsers(dest="stage", required=True)

    sub.add_parser("browser", help="Playwright/browser-pool sanity, no network")

    p_login = sub.add_parser("login", help="seed sessions against real sites and screenshot them")
    p_login.add_argument("--sites", nargs="*", default=None)

    p_replay = sub.add_parser("verify-replay", help="score synthetic gold trajectories through verify()")
    p_replay.add_argument("--data", default="resources_servers/webarena/data/example.jsonl")
    p_replay.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()
    stage_fn = {
        "browser": stage_browser,
        "login": stage_login,
        "verify-replay": stage_verify_replay,
    }[args.stage]
    return asyncio.run(stage_fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
