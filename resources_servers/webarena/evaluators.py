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
"""Pure-function WebArena evaluators (string_match / url_match / program_html scoring).

Ported from the internal WebArena harness (osworld_internal
webarena/common/classic_evaluation.py) so rewards computed here match the
standalone eval harness. Everything in this module is stdlib-only and
side-effect free: LLM-judge calls are returned as pending request payloads for
the resources server to resolve, and program_html DOM extraction happens in
app.py (this module only scores the extracted content).
"""

from __future__ import annotations

import html
import json
import re
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple


########################################
# Site URL placeholders
########################################

PLACEHOLDER_TO_SITE = {
    "__GITLAB__": "gitlab",
    "__REDDIT__": "reddit",
    "__SHOPPING__": "shopping",
    "__SHOPPING_ADMIN__": "shopping_admin",
    "__WIKIPEDIA__": "wikipedia",
    "__MAP__": "map",
    "__CLASSIFIEDS__": "classifieds",
}


def substitute_site_placeholders(text: Any, site_urls: Dict[str, str]) -> str:
    """Replace __SHOPPING__-style placeholders with configured site base URLs."""
    result = "" if text is None else str(text)
    for placeholder, site in PLACEHOLDER_TO_SITE.items():
        url = site_urls.get(site)
        if url:
            result = result.replace(placeholder, url.rstrip("/"))
    return result


########################################
# string_match primitives
########################################


def clean_answer(answer: Any) -> str:
    text = "" if answer is None else str(answer)
    text = text.strip()
    if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
        text = text[1:-1]
    return text.lower()


def exact_match(ref: Any, pred: Any) -> float:
    return float(clean_answer(pred) == clean_answer(ref))


def word_tokenize_like_webarena(text: str) -> List[str]:
    """Small tokenizer for the original single-character must_include guard."""
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def must_include(ref: Any, pred: Any, tokenize: bool = False) -> float:
    clean_ref = clean_answer(ref)
    clean_pred = clean_answer(pred)
    if tokenize and len(clean_ref) == 1 and len(word_tokenize_like_webarena(clean_ref)) == 1:
        return float(clean_ref in word_tokenize_like_webarena(clean_pred))
    return float(clean_ref in clean_pred)


def reference_alternatives(ref: Any) -> List[str]:
    if isinstance(ref, list):
        return [str(item) for item in ref]
    return [str(ref)]


def format_required_group(alternatives: List[str]) -> str:
    if len(alternatives) == 1:
        return alternatives[0]
    return "one of: " + " or ".join(alternatives)


########################################
# LLM-judge request payloads
#
# string_match_local() cannot call the judge itself (that requires the
# server_client). Instead it returns JudgeRequest dicts; the resources server
# resolves each one against its configured judge model server and multiplies
# the returned 0/1 into the score.
########################################

JUDGE_PASS_LABELS = {
    "fuzzy_match": "correct",
    "exact_match_fallback": "correct",
    "must_include_fallback": "correct",
    "ua_match": "same",
}

JUDGE_ALLOWED_LABELS = {
    "fuzzy_match": {"correct", "incorrect", "partially correct"},
    "exact_match_fallback": {"correct", "incorrect", "partially correct"},
    "must_include_fallback": {"correct", "incorrect", "partially correct"},
    "ua_match": {"same", "different"},
}

JUDGE_SYSTEM_MESSAGE = "You are a helpful assistant"


def build_fuzzy_match_user_message(question: str, reference: str, pred: str) -> str:
    return (
        "Help a teacher grade the answer of a student given a question. "
        "The goal is to evaluate whether the student's answer is semantically "
        "equivalent to the reference answer.\n\n"
        "Rules:\n"
        "- The student's answer is exactly the text inside <student_answer> tags.\n"
        "- Do not infer, complete, rewrite, or invent the student's answer.\n"
        "- If <student_answer> is empty, whitespace only, or missing, use incorrect.\n"
        "- Use correct only when the student's answer is semantically equivalent "
        "to the reference answer.\n"
        "- Use partially correct when the answer contains some but not all required meaning.\n"
        "- Provide brief reasoning grounded only in the student's answer and reference.\n\n"
        f"question: {question}\n"
        f"reference answer: {reference}\n"
        "all the string 'N/A' that you see is a special sequence that means 'not achievable'\n"
        f"<student_answer>\n{pred}\n</student_answer>\n\n"
        'Return JSON only: {"judgement": "correct" | "incorrect" | '
        '"partially correct", "reasoning": "..."}'
    )


def build_must_include_user_message(question: str, reference: str, pred: str) -> str:
    return (
        "Help a teacher to grade whether a student's answer is correct for a web "
        "task. The student may use different phrasing or wording. The answer is "
        "correct if it semantically contains every required item below. For a "
        "required line that starts with 'one of:', the answer only needs to "
        "contain one option from that line.\n\n"
        "Rules:\n"
        "- The student's answer is exactly the text inside <student_answer> tags.\n"
        "- Do not infer, complete, rewrite, or invent the student's answer.\n"
        "- If <student_answer> is empty, whitespace only, or missing, use incorrect.\n"
        "- Use correct only if every required line is satisfied by the student's answer.\n"
        "- Use partially correct when only some required lines are satisfied.\n"
        "- Provide brief reasoning grounded only in the student's answer and required information.\n\n"
        f"question: {question}\n"
        f"required information:\n{reference}\n"
        f"<student_answer>\n{pred}\n</student_answer>\n\n"
        'Return JSON only: {"judgement": "correct" | "incorrect" | '
        '"partially correct", "reasoning": "..."}'
    )


def build_ua_match_user_message(question: str, reference: str, pred: str) -> str:
    return (
        f"task: {question}\n"
        f"actual unachievable reason: {reference}\n"
        f"<reported_unachievable_reason>\n{pred}\n</reported_unachievable_reason>\n"
        "The task described above is inherently unachievable due to the reason "
        "specified under 'actual unachievable reason'. An individual previously "
        "attempted this task and was unable to complete it. They provided a "
        "reason for their failure, which is exactly the text inside "
        "<reported_unachievable_reason> tags. Do not infer, complete, rewrite, "
        "or invent the reported reason. If the reported reason is empty, "
        "whitespace only, or missing, use different. Determine if the reported "
        "reason aligns with the actual reason, even if implicitly. Provide brief "
        "reasoning grounded only in the reported and actual reasons.\n\n"
        'Return JSON only: {"judgement": "same" | "different", "reasoning": "..."}'
    )


_JUDGE_MESSAGE_BUILDERS = {
    "fuzzy_match": build_fuzzy_match_user_message,
    "exact_match_fallback": build_fuzzy_match_user_message,
    "must_include_fallback": build_must_include_user_message,
    "ua_match": build_ua_match_user_message,
}


def make_judge_request(judge_type: str, question: str, reference: str, pred: str) -> Dict[str, str]:
    return {
        "judge_type": judge_type,
        "question": question,
        "reference": reference,
        "prediction": pred,
        "user_message": _JUDGE_MESSAGE_BUILDERS[judge_type](question, reference, pred),
    }


def parse_judge_json_label(response_text: str, allowed_labels: set[str]) -> Optional[str]:
    """Extract the "judgement" label from a judge response that should be JSON.

    Tolerates code fences, surrounding prose, and thinking blocks: takes the
    last JSON object with a "judgement" key; falls back to a regex scan.
    """
    text = response_text or ""
    text = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text, flags=re.DOTALL)

    candidates = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
    for candidate in reversed(candidates):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        label = str(obj.get("judgement", "")).strip().lower()
        if label in allowed_labels:
            return label

    match = re.findall(r'"judgement"\s*:\s*"([^"]+)"', text)
    for label in reversed(match):
        label = label.strip().lower()
        if label in allowed_labels:
            return label
    return None


def judge_passed(judge_type: str, response_text: str) -> float:
    label = parse_judge_json_label(response_text, JUDGE_ALLOWED_LABELS[judge_type])
    return float(label == JUDGE_PASS_LABELS[judge_type])


########################################
# string_match
########################################


def string_match_local(eval_cfg: Dict[str, Any], intent: str, answer: Any) -> Tuple[float, List[Dict[str, str]]]:
    """Local (non-LLM) portion of the classic string_match evaluator.

    Returns (local_score, pending_judge_requests). The final score is
    local_score multiplied by the 0/1 outcome of every pending judge request;
    with no judge configured, pending requests score 0.0 (conservative).
    """
    pred = clean_answer(answer)
    refs = eval_cfg.get("reference_answers") or {}
    score = 1.0
    pending: List[Dict[str, str]] = []

    for approach, value in refs.items():
        if approach == "exact_match":
            alternatives = reference_alternatives(value)
            cur_score = max(exact_match(ref=alt, pred=pred) for alt in alternatives)
            if cur_score != 1.0:
                pending.append(
                    make_judge_request(
                        "exact_match_fallback",
                        question=intent,
                        reference=format_required_group(alternatives),
                        pred=pred,
                    )
                )
            else:
                score *= cur_score
        elif approach == "must_include":
            required_groups = []
            rule_score = 1.0
            for must_value in value:
                alternatives = reference_alternatives(must_value)
                required_groups.append(alternatives)
                rule_score *= max(
                    must_include(ref=alt, pred=pred, tokenize=(len(value) == 1)) for alt in alternatives
                )
            if rule_score == 1.0:
                score *= rule_score
            else:
                reference = "\n".join(
                    f"{idx}. {format_required_group(alternatives)}"
                    for idx, alternatives in enumerate(required_groups, start=1)
                )
                pending.append(
                    make_judge_request("must_include_fallback", question=intent, reference=reference, pred=pred)
                )
        elif approach == "fuzzy_match":
            if value == "N/A":
                if exact_match(ref=value, pred=pred) != 1.0:
                    pending.append(
                        make_judge_request(
                            "ua_match",
                            question=intent,
                            reference=str(eval_cfg.get("string_note", "")),
                            pred=pred,
                        )
                    )
            else:
                for reference in value:
                    pending.append(
                        make_judge_request("fuzzy_match", question=intent, reference=str(reference), pred=pred)
                    )
        else:
            raise ValueError(f"Unknown string_match approach: {approach}")

    return score, pending


########################################
# url_match
########################################


def clean_url(url: Any) -> str:
    return urllib.parse.urldefrag(str(url).rstrip("/")).url


def reference_url_alternatives(reference_url: Any) -> List[str]:
    if isinstance(reference_url, list):
        candidates = reference_url
    else:
        candidates = str(reference_url or "").split(" |OR| ")

    urls: List[str] = []
    for candidate in candidates:
        for url in str(candidate).split(" |OR| "):
            url = url.strip()
            if url:
                urls.append(url)
    return urls


def parse_url(url: str) -> Tuple[str, Dict[str, List[str]]]:
    parsed_url = urllib.parse.urlparse(url)
    base_path = parsed_url.netloc + parsed_url.path
    query = urllib.parse.parse_qs(parsed_url.query)
    return base_path, query


def url_match(eval_cfg: Dict[str, Any], current_url: str, site_urls: Dict[str, str]) -> float:
    """Classic WebArena URL matching (rule: "GOLD in PRED")."""
    pred = clean_url(current_url)
    ref_urls = [
        clean_url(substitute_site_placeholders(url, site_urls))
        for url in reference_url_alternatives(eval_cfg.get("reference_url"))
    ]
    if not ref_urls:
        return 0.0

    matching_rule = eval_cfg.get("url_note", "GOLD in PRED")
    if matching_rule != "GOLD in PRED":
        raise ValueError(f"Unknown URL matching rule: {matching_rule}")

    pred_base_path, pred_query = parse_url(pred)
    for ref_url in ref_urls:
        ref_base_path, ref_query = parse_url(ref_url)
        if ref_base_path not in pred_base_path:
            continue
        if all(
            any(ref_value in pred_value for pred_value in pred_query.get(key, []) for ref_value in ref_values)
            for key, ref_values in ref_query.items()
        ):
            return 1.0
    return 0.0


def score_url_match_candidates(
    eval_cfg: Dict[str, Any], candidate_urls: List[str], site_urls: Dict[str, str]
) -> Tuple[float, str, List[str]]:
    """Best score over all candidate URLs (open tabs / trajectory tail)."""
    unique_urls = list(dict.fromkeys(candidate_urls))
    if not unique_urls:
        return 0.0, "", []
    for candidate_url in unique_urls:
        if url_match(eval_cfg, candidate_url, site_urls):
            return 1.0, candidate_url, unique_urls
    return 0.0, unique_urls[0], unique_urls


########################################
# program_html content scoring
########################################


def score_program_html_required(required: Dict[str, Any], selected_element: Any) -> float:
    score = 1.0
    selected_element = html.unescape(str(selected_element))
    if "exact_match" in required:
        score *= exact_match(ref=required["exact_match"], pred=selected_element)
    elif "must_include" in required:
        for content in required["must_include"]:
            content_or = str(content).split(" |OR| ")
            score *= float(any(must_include(ref=part, pred=selected_element) for part in content_or))
    else:
        raise ValueError(f"Unknown required_contents: {required.keys()}")
    return score


########################################
# Trajectory helpers
########################################


def trajectory_candidate_urls(step_urls: List[str]) -> List[str]:
    """Unique non-empty URLs from a trajectory, most recent first."""
    candidates: List[str] = []
    for url in reversed(step_urls):
        if url and not url.startswith("error:") and url not in candidates:
            candidates.append(url)
    return candidates
