from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional, Sequence

import requests

# Allow importing evaluation helpers when script is executed from repo root.
SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from evaluation.dataset import (
    QuestionSample,
    load_question_prompts,
    load_supabase_qa_export,
    merge_unique_questions,
)


def _resolve_backend_url(cli_url: Optional[str]) -> str:
    if cli_url:
        return cli_url
    env_url = os.getenv("EVAL_BACKEND_URL") or os.getenv("BACKEND_URL")
    if env_url:
        return env_url
    return "http://localhost:8000"


def _iter_questions(samples: Iterable[QuestionSample], prompts: Iterable[str]) -> list[str]:
    from evaluation.dataset import merge_unique_questions

    logged_questions = [sample.question for sample in samples]
    prompt_questions = list(prompts)
    return merge_unique_questions(logged_questions, prompt_questions)


def evaluate(
    backend_url: str,
    questions: list[str],
    *,
    limit: Optional[int] = None,
    timeout: float = 20.0,
) -> dict:
    endpoint = backend_url.rstrip("/") + "/qa"
    payloads = []
    for idx, question in enumerate(questions, start=1):
        if limit and idx > limit:
            break
        payloads.append(question)

    results = []
    latencies = []
    for question in payloads:
        try:
            response = requests.post(
                endpoint,
                json={"question": question},
                timeout=timeout,
            )
            latency_ms = response.elapsed.total_seconds() * 1000.0
            latencies.append(latency_ms)
            data = response.json()
        except Exception as exc:  # noqa: BLE001 - we want to capture every failure
            results.append(
                {
                    "question": question,
                    "status": "error",
                    "error": str(exc),
                }
            )
            continue

        if not isinstance(data, dict):
            results.append(
                {
                    "question": question,
                    "status": "error",
                    "error": f"Unexpected response: {data!r}",
                }
            )
            continue

        results.append(
            {
                "question": question,
                "status": data.get("status"),
                "latency_ms": latency_ms,
                "sources": data.get("sources"),
            }
        )

    answered = sum(1 for item in results if item.get("status") == "answered")
    escalated = sum(1 for item in results if item.get("status") == "escalated")
    suggested = sum(1 for item in results if item.get("status") == "suggested")
    errors = sum(1 for item in results if item.get("status") == "error")

    status_counts = Counter(item.get("status") or "unknown" for item in results)
    source_counts = Counter()
    for item in results:
        for source in item.get("sources") or []:
            doc_id = (source or {}).get("document_id")
            if isinstance(doc_id, str):
                if doc_id.startswith("calendar:"):
                    source_counts["calendar"] += 1
                elif doc_id.startswith("manual:"):
                    source_counts["manual"] += 1
                else:
                    source_counts["documents"] += 1

    metrics = {
        "total": len(results),
        "answered": answered,
        "escalated": escalated,
        "suggested": suggested,
        "errors": errors,
        "latency_ms_avg": statistics.mean(latencies) if latencies else None,
        "latency_ms_p95": statistics.quantiles(latencies, n=20)[-1] if len(latencies) >= 20 else None,
        "details": results,
        "status_counts": dict(status_counts),
        "source_counts": dict(source_counts),
    }
    return metrics


def _load_expectations(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError("Expectations file must contain a JSON list.")
        return data


def _status_satisfies(expected: str, actual: Optional[str]) -> bool:
    if actual is None:
        return False
    if expected == "unsupported":
        return actual not in {"answered", "suggested"}
    if expected == "answered":
        return actual == "answered"
    if expected == "escalated":
        return actual == "escalated"
    if expected == "error":
        return actual == "error"
    return actual == expected


def evaluate_expectations(results: list[dict], expectations_path: Path) -> list[dict]:
    expectations = _load_expectations(expectations_path)
    result_map = {item.get("question", "").strip().lower(): item for item in results}
    mismatches: list[dict] = []

    for expectation in expectations:
        question = expectation.get("question")
        if not question:
            continue
        key = question.strip().lower()
        result = result_map.get(key)
        if result is None:
            mismatches.append(
                {
                    "question": question,
                    "reason": "missing_result",
                }
            )
            continue

        issues: list[str] = []
        expected_status = expectation.get("expected_status")
        if expected_status:
            if not _status_satisfies(expected_status, result.get("status")):
                issues.append(
                    f"expected status {expected_status!r}, got {result.get('status')!r}"
                )

        required_ids = expectation.get("required_document_ids") or []
        if required_ids:
            actual_ids = {
                (src or {}).get("document_id")
                for src in result.get("sources") or []
                if isinstance(src, dict)
            }
            missing = [doc_id for doc_id in required_ids if doc_id not in actual_ids]
            if missing:
                issues.append(f"missing required documents: {missing}")

        forbidden_ids = expectation.get("forbidden_document_ids") or []
        if forbidden_ids:
            actual_ids = {
                (src or {}).get("document_id")
                for src in result.get("sources") or []
                if isinstance(src, dict)
            }
            present = [doc_id for doc_id in forbidden_ids if doc_id in actual_ids]
            if present:
                issues.append(f"forbidden documents present: {present}")

        if issues:
            mismatches.append(
                {
                    "question": question,
                    "issues": issues,
                }
            )

    return mismatches


def _load_extra_questions(paths: Sequence[Path]) -> list[str]:
    extra_questions: list[str] = []
    for path in paths:
        if not path or not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                extra_questions.extend(str(item).strip() for item in data if str(item).strip())
        except Exception:
            continue
    return extra_questions


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run baseline evaluation against the QA endpoint.")
    parser.add_argument(
        "--backend-url",
        help="HTTP base URL for the QA backend (default: env EVAL_BACKEND_URL or http://localhost:8000).",
    )
    parser.add_argument(
        "--log-path",
        default=str(Path(__file__).resolve().parents[2] / "data" / "Supabase Snippet Ad hoc queries.csv"),
        help="Path to the Supabase QA export CSV.",
    )
    parser.add_argument(
        "--questions-path",
        default=str(Path(__file__).resolve().parents[2] / "data" / "Supabase Snippet Questions asked.csv"),
        help="Path to the question-only CSV.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Optional limit on the number of questions evaluated.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="HTTP timeout in seconds for each question.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the raw results JSON for further analysis.",
    )
    parser.add_argument(
        "--expectations",
        type=Path,
        help="Optional JSON file with expected outcomes for specific questions.",
    )
    parser.add_argument(
        "--extra-questions",
        type=Path,
        action="append",
        help="Optional JSON file containing additional questions to include in the run (list of strings).",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Do not exit with a non-zero status when mismatches or errors are encountered.",
    )

    args = parser.parse_args(argv)

    backend_url = _resolve_backend_url(args.backend_url)

    samples = load_supabase_qa_export(Path(args.log_path), limit=args.limit)
    prompts = load_question_prompts(Path(args.questions_path), limit=args.limit)
    questions = _iter_questions(samples, prompts)

    manual_questions = []
    default_manual = SCRIPT_ROOT / "evaluation" / "manual_questions.json"
    question_paths: list[Path] = []
    if default_manual.exists():
        question_paths.append(default_manual)
    if args.extra_questions:
        question_paths.extend(args.extra_questions)
    manual_questions = _load_extra_questions(question_paths)

    if manual_questions:
        questions = merge_unique_questions(questions, manual_questions)

    metrics = evaluate(
        backend_url,
        questions,
        limit=args.limit,
        timeout=args.timeout,
    )

    if args.expectations:
        mismatches = evaluate_expectations(metrics["details"], args.expectations)
        metrics["expectation_mismatches"] = mismatches
        metrics["expectations_file"] = str(args.expectations)
        metrics["expectations_passed"] = not mismatches
    else:
        metrics["expectations_passed"] = None

    print(json.dumps(metrics, indent=2, sort_keys=True))

    if args.output:
        args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if not args.no_fail:
        if metrics.get("errors"):
            return 1
        mismatches = metrics.get("expectation_mismatches")
        if isinstance(mismatches, list) and mismatches:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
