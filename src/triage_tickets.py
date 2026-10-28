"""Privacy-preserving support-ticket triage using the supplied offline LLM mock."""
from __future__ import annotations

import csv
import json
import logging
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import mock_llm


ROOT = Path(__file__).resolve().parents[1]
TICKETS_PATH = ROOT / "data" / "raw" / "tickets.csv"
SEGMENTS_PATH = ROOT / "outputs" / "customer_segments.csv"
OUTPUTS_DIR = ROOT / "outputs"

MAX_INPUT_TOKENS = 400
TOKEN_BUDGET = 100_000
DEMO_TOKEN_BUDGET = 6_000
BASE_DELAY = 0.05
MAX_RETRIES = 3
ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}

REDACTION_PATTERNS = (
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    (re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    (re.compile(r"ORD-\d{6}"), "[ORDER_ID]"),
)
SHORT_ORDER_ID_PATTERN = re.compile(r"ORD-\d{3,5}\b")
ANY_ORDER_ID_PATTERN = re.compile(r"\bORD-\d+\b")

ZERO_SHOT_PROMPT = """CLASSIFY the customer's support ticket into exactly one category:
billing, delivery, technical, or fraud.
Reply with JSON only, in this form: {"category": "...", "reason": "..."}."""

FEW_SHOT_PROMPT = """CLASSIFY the customer's support ticket into exactly one category:
billing, delivery, technical, or fraud.
Reply with JSON only, in this form: {"category": "...", "reason": "..."}.
EXAMPLES:
Ticket: I was charged twice for my purchase.
Reply: {"category": "billing", "reason": "duplicate charge"}
Ticket: Someone stole my account and made an unauthorized payment.
Reply: {"category": "fraud", "reason": "unauthorized account activity"}
Ticket: The delivery arrived late.
Reply: {"category": "delivery", "reason": "late delivery"}"""

LOGGER = logging.getLogger("ticket_triage")


def redact_tickets(
    tickets: list[dict[str, str]],
) -> tuple[list[dict[str, str]], Counter[str]]:
    redacted_tickets: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    count_names = ("emails", "phones", "order_ids")

    for ticket in tickets:
        safe_ticket = dict(ticket)
        text = ticket["text"]
        for (pattern, replacement), count_name in zip(REDACTION_PATTERNS, count_names):
            text, count = pattern.subn(replacement, text)
            counts[count_name] += count
            if count_name == "order_ids":
                counts["six_digit_order_ids"] += count
        text, fallback_count = SHORT_ORDER_ID_PATTERN.subn("[ORDER_ID]", text)
        counts["short_order_ids"] += fallback_count
        counts["order_ids"] += fallback_count
        safe_ticket["text"] = text
        for pattern, _ in REDACTION_PATTERNS:
            assert pattern.search(text) is None, (
                f"Unredacted personal information remains in {ticket['ticket_id']}"
            )
        assert ANY_ORDER_ID_PATTERN.search(text) is None, (
            f"Unredacted order ID remains in {ticket['ticket_id']}"
        )
        redacted_tickets.append(safe_ticket)

    counts["total"] = sum(counts[name] for name in count_names)
    return redacted_tickets, counts


def read_tickets() -> list[dict[str, str]]:
    with TICKETS_PATH.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def read_segments() -> dict[str, dict[str, str]]:
    with SEGMENTS_PATH.open(encoding="utf-8", newline="") as source:
        return {row["customer_id"]: row for row in csv.DictReader(source)}


def call_with_retry(
    ticket_id: str, messages: list[dict[str, str]]
) -> tuple[Any, int]:
    retries = 0
    for retry_number in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages), retries
        except mock_llm.TransientLLMError:
            if retry_number == MAX_RETRIES:
                raise
            wait = BASE_DELAY * (2**retry_number) * random.uniform(0.9, 1.15)
            LOGGER.warning(
                "Retry ticket=%s retry=%d wait=%.4fs",
                ticket_id,
                retry_number,
                wait,
            )
            retries += 1
            time.sleep(wait)
    raise RuntimeError(f"Retry loop exited unexpectedly for {ticket_id}")


def validate_reply(reply: str) -> tuple[str | None, str | None]:
    try:
        parsed = json.loads(reply)
    except json.JSONDecodeError:
        words = re.findall(r"[a-z]+", reply.lower())
        matches = [word for word in words if word in ALLOWED_CATEGORIES]
        if len(matches) == 1:
            return matches[0], "classified_keyword"
        return None, "flagged_invalid_output"

    if isinstance(parsed, dict) and parsed.get("category") in ALLOWED_CATEGORIES:
        return parsed["category"], "classified_json"
    return None, "flagged_invalid_output"


def run_pipeline(
    tickets: list[dict[str, str]],
    system_prompt: str,
    token_budget: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    mock_llm.reset()
    results: list[dict[str, Any]] = []
    running_tokens = 0
    retry_count = 0

    for ticket in tickets:
        result: dict[str, Any] = {
            "ticket_id": ticket["ticket_id"],
            "customer_id": ticket["customer_id"],
            "status": "",
            "category": "",
            "tokens_used": 0,
        }
        text = ticket["text"]
        if not text.strip():
            result["status"] = "skipped_empty"
            results.append(result)
            continue
        if mock_llm.est_tokens(text) > MAX_INPUT_TOKENS:
            result["status"] = "flagged_long"
            results.append(result)
            continue
        if token_budget is not None and running_tokens >= token_budget:
            result["status"] = "budget_exhausted"
            results.append(result)
            continue

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]
        assert messages[-1]["content"] == text
        for message in messages:
            for pattern, _ in REDACTION_PATTERNS:
                assert pattern.search(message["content"]) is None, (
                    f"Personal information would be sent for {ticket['ticket_id']}"
                )
            assert ANY_ORDER_ID_PATTERN.search(message["content"]) is None, (
                f"An order ID would be sent for {ticket['ticket_id']}"
            )

        response, retries = call_with_retry(ticket["ticket_id"], messages)
        retry_count += retries
        used_tokens = response.usage.total_tokens
        running_tokens += used_tokens
        result["tokens_used"] = used_tokens
        reply = response.choices[0].message.content or ""
        category, status = validate_reply(reply)
        result["status"] = status
        if category is not None:
            result["category"] = category
        results.append(result)

    return results, retry_count


def report_redactions(counts: Counter[str]) -> None:
    print(
        "Redactions: "
        f"emails={counts['emails']}, phones={counts['phones']}, "
        f"order_ids={counts['order_ids']} "
        f"(six_digit={counts['six_digit_order_ids']}, "
        f"short_id_fallback={counts['short_order_ids']}), total={counts['total']}; "
        "remaining matches=0 (asserted for every redacted ticket)"
    )


def report_run(
    label: str,
    tickets: list[dict[str, str]],
    results: list[dict[str, Any]],
    retry_count: int,
) -> None:
    original_by_id = {ticket["ticket_id"]: ticket for ticket in tickets}
    classified = [
        result
        for result in results
        if result["status"] in {"classified_json", "classified_keyword"}
    ]
    correct = sum(
        result["category"] == original_by_id[result["ticket_id"]]["true_category"]
        for result in classified
    )
    total_tokens = sum(result["tokens_used"] for result in results)
    accuracy = correct / len(classified) if classified else 0.0
    tokens_per_classified = total_tokens / len(classified) if classified else 0.0
    print(
        f"{label}: classified={len(classified)}, accuracy={accuracy:.1%} "
        f"({correct}/{len(classified)}), total_tokens={total_tokens}, "
        f"tokens_per_classified_ticket={tokens_per_classified:.2f}, "
        f"retries={retry_count}"
    )
    print(f"{label} statuses: {dict(sorted(Counter(r['status'] for r in results).items()))}")

    if label == "zero-shot":
        confusion = {
            actual: Counter(
                result["category"]
                for result in classified
                if original_by_id[result["ticket_id"]]["true_category"] == actual
            )
            for actual in sorted(ALLOWED_CATEGORIES)
        }
        print("Zero-shot confusion matrix (rows=true, columns=predicted):")
        print("true_category," + ",".join(sorted(ALLOWED_CATEGORIES)))
        for actual in sorted(ALLOWED_CATEGORIES):
            print(
                actual
                + ","
                + ",".join(str(confusion[actual][pred]) for pred in sorted(ALLOWED_CATEGORIES))
            )


def route_results(
    tickets: list[dict[str, str]],
    results: list[dict[str, Any]],
    segments: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    routed: list[dict[str, Any]] = []
    human_statuses = {
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted",
    }
    for result in results:
        customer_id = result["customer_id"]
        segment = segments.get(customer_id, {})
        status = result["status"]
        category = result["category"]
        if status in human_statuses:
            route, priority = "human_review", "NORMAL"
        elif category == "fraud":
            route, priority = "security_team", "HIGH"
        elif segment.get("is_premium", "").strip().lower() == "true":
            route, priority = "premium_desk", "HIGH"
        else:
            route, priority = "support_queue", "NORMAL"
        routed.append(
            {
                "ticket_id": result["ticket_id"],
                "customer_id": customer_id,
                "status": status,
                "category": category,
                "segment_name": segment.get("segment_name", ""),
                "priority": priority,
                "route": route,
                "tokens_used": result["tokens_used"],
            }
        )
    return routed


def write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "ticket_id",
        "customer_id",
        "status",
        "category",
        "segment_name",
        "priority",
        "route",
        "tokens_used",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    raw_tickets = read_tickets()
    tickets, redaction_counts = redact_tickets(raw_tickets)
    report_redactions(redaction_counts)

    empty_count = sum(not ticket["text"].strip() for ticket in tickets)
    long_tickets = [
        ticket
        for ticket in tickets
        if ticket["text"].strip() and mock_llm.est_tokens(ticket["text"]) > MAX_INPUT_TOKENS
    ]
    assert len(tickets) == 164
    assert empty_count == 4
    assert len(long_tickets) == 4
    assert all(mock_llm.est_tokens(ticket["text"]) == 734 for ticket in long_tickets)

    zero_results, zero_retries = run_pipeline(tickets, ZERO_SHOT_PROMPT)
    report_run("zero-shot", raw_tickets, zero_results, zero_retries)
    few_results, few_retries = run_pipeline(tickets, FEW_SHOT_PROMPT)
    report_run("few-shot", raw_tickets, few_results, few_retries)
    assert zero_retries == few_retries == 19
    assert Counter(result["status"] for result in zero_results) == {
        "classified_json": 135,
        "classified_keyword": 12,
        "flagged_invalid_output": 9,
        "flagged_long": 4,
        "skipped_empty": 4,
    }
    assert Counter(result["status"] for result in few_results) == Counter(
        result["status"] for result in zero_results
    )

    budget_results, budget_retries = run_pipeline(
        tickets, ZERO_SHOT_PROMPT, token_budget=DEMO_TOKEN_BUDGET
    )
    exhausted_count = sum(row["status"] == "budget_exhausted" for row in budget_results)
    if exhausted_count == 0:
        raise AssertionError(
            f"No tickets exhausted the demo token budget of {DEMO_TOKEN_BUDGET}"
        )
    print(
        f"Budget demo: budget={DEMO_TOKEN_BUDGET}, exhausted={exhausted_count}, "
        f"retries={budget_retries}"
    )
    demo_rows = route_results(tickets, budget_results, read_segments())
    write_results(OUTPUTS_DIR / "triage_budget_demo.csv", demo_rows)

    final_results, final_retries = run_pipeline(
        tickets, FEW_SHOT_PROMPT, token_budget=TOKEN_BUDGET
    )
    final_rows = route_results(tickets, final_results, read_segments())
    write_results(OUTPUTS_DIR / "triage_results.csv", final_rows)
    assert len(final_rows) == 164
    print(
        f"Final few-shot output: budget={TOKEN_BUDGET}, rows={len(final_rows)}, "
        f"retries={final_retries}, file=outputs/triage_results.csv"
    )


if __name__ == "__main__":
    main()
