"""Retrieval-grounded policy answers and a resumable refund agent."""
from __future__ import annotations

import csv
import json
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm


ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "data" / "policies"
DATABASE_PATH = ROOT / "data" / "instamart.db"
POLICY_EVAL_PATH = ROOT / "data" / "raw" / "policy_eval.csv"
SCENARIOS_PATH = ROOT / "data" / "raw" / "agent_scenarios.csv"
OUTPUTS_DIR = ROOT / "outputs"
STATE_DIR = ROOT / "state"
LEDGER_PATH = OUTPUTS_DIR / "refund_ledger.jsonl"
AGENT_RUNS_PATH = OUTPUTS_DIR / "agent_runs.jsonl"

CHUNK_SIZE = 40
CHUNK_OVERLAP = 10


def chunk_documents(overlap: int = CHUNK_OVERLAP) -> list[dict[str, Any]]:
    if not 0 <= overlap < CHUNK_SIZE:
        raise ValueError("overlap must be between 0 and 39 words")
    chunks: list[dict[str, Any]] = []
    step = CHUNK_SIZE - overlap
    for path in sorted(POLICY_DIR.glob("*.txt")):
        words = path.read_text(encoding="utf-8").split()
        for start in range(0, len(words), step):
            chunk_words = words[start : start + CHUNK_SIZE]
            if not chunk_words:
                break
            chunks.append({"source": path.name, "text": " ".join(chunk_words)})
            if start + CHUNK_SIZE >= len(words):
                break
    return chunks


CHUNKS = chunk_documents()
VECTORIZER = TfidfVectorizer()
CHUNK_MATRIX = VECTORIZER.fit_transform([chunk["text"] for chunk in CHUNKS])


def retrieve(
    query: str, k: int = 3, overlap: int = CHUNK_OVERLAP
) -> list[dict[str, Any]]:
    chunks = CHUNKS if overlap == CHUNK_OVERLAP else chunk_documents(overlap)
    if not query.strip() or not chunks or k <= 0:
        return []
    if chunks is CHUNKS:
        vectorizer = VECTORIZER
        matrix = CHUNK_MATRIX
    else:
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform([chunk["text"] for chunk in chunks])
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()
    ranked = sorted(range(len(chunks)), key=lambda index: (-scores[index], index))
    return [
        {
            "source": chunks[index]["source"],
            "score": float(scores[index]),
            "text": chunks[index]["text"],
        }
        for index in ranked[: min(k, len(chunks))]
    ]


POLICY_SYSTEM_PROMPT = (
    "POLICY ASSISTANT: Answer only from the supplied policy context. "
    "If the context does not contain the answer, say exactly: "
    "I could not find that information in the policy documents."
)


def _policy_user_message(
    question: str, retrieved: list[dict[str, Any]]
) -> str:
    context = "\n".join(
        f"[{chunk['source']}]\n{chunk['text']}" for chunk in retrieved
    )
    return f"CONTEXT:\n{context}\nQUESTION:\n{question}"


def answer(
    question: str, overlap: int = CHUNK_OVERLAP
) -> tuple[str, list[dict[str, Any]]]:
    sources = retrieve(question, k=3, overlap=overlap)
    response = mock_llm.chat(
        [
            {"role": "system", "content": POLICY_SYSTEM_PROMPT},
            {"role": "user", "content": _policy_user_message(question, sources)},
        ]
    )
    return response.choices[0].message.content or "", sources


def evaluate_policy(overlap: int = CHUNK_OVERLAP) -> dict[str, int]:
    with POLICY_EVAL_PATH.open(encoding="utf-8-sig", newline="") as source:
        questions = list(csv.DictReader(source))
    answerable = [
        row for row in questions if row["expected_doc"].strip().upper() != "NONE"
    ]
    unanswerable = [
        row for row in questions if row["expected_doc"].strip().upper() == "NONE"
    ]
    retrieved_expected = 0
    grounded_expected = 0
    fallback_unanswerable = 0
    for row in answerable:
        sources = retrieve(row["question"], k=3, overlap=overlap)
        retrieved_expected += any(
            chunk["source"] == row["expected_doc"] for chunk in sources
        )
        response, _ = answer(row["question"], overlap=overlap)
        grounded_expected += row["expected_fact"] in response
    for row in unanswerable:
        response, _ = answer(row["question"], overlap=overlap)
        fallback_unanswerable += response.startswith("I could not find")
    return {
        "answerable": len(answerable),
        "retrieved_expected": retrieved_expected,
        "grounded_expected": grounded_expected,
        "unanswerable": len(unanswerable),
        "fallback_unanswerable": fallback_unanswerable,
    }


def get_order(order_id: str) -> dict[str, Any]:
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT order_id, status, order_amount, delivery_minutes "
            "FROM orders WHERE order_id = ?",
            (order_id,),
        ).fetchone()
        if row is None and order_id.startswith("ORD-"):
            # mock_llm recognizes six-digit identifiers; resolve its zero-padded alias.
            digits = order_id[4:]
            canonical = digits.lstrip("0") or "0"
            row = connection.execute(
                "SELECT order_id, status, order_amount, delivery_minutes "
                "FROM orders WHERE order_id = ?",
                (f"ORD-{canonical}",),
            ).fetchone()
    if row is None:
        return {"error": "order not found"}
    return {
        "order_id": row["order_id"],
        "status": row["status"],
        "order_amount": float(row["order_amount"]),
        "delivery_minutes": (
            float(row["delivery_minutes"])
            if row["delivery_minutes"] is not None
            else None
        ),
    }


def search_policy(query: str) -> dict[str, Any]:
    return {"results": retrieve(query, k=3)}


def calculate_refund(
    order_amount: float, delivery_minutes: float | None
) -> dict[str, float]:
    amount = float(order_amount) if delivery_minutes is not None and delivery_minutes > 45 else 0.0
    return {"refund_inr": round(amount, 2)}


def _write_ledger(
    tool_name: str, order_id: str, refund_inr: float
) -> dict[str, float]:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2),
    }
    with LEDGER_PATH.open("a", encoding="utf-8") as destination:
        destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"refund_inr": row["refund_inr"]}


def _validate_refund_action(
    order_id: str, refund_inr: float, escalate: bool
) -> str:
    order = get_order(order_id)
    if "error" in order:
        raise ValueError(f"Cannot process refund: {order['error']}")
    if order["status"] != "completed":
        raise ValueError("Only completed orders can be refunded")
    if order["delivery_minutes"] is None or order["delivery_minutes"] <= 45:
        raise ValueError("A late-delivery refund requires delivery after 45 minutes")
    expected_refund = round(float(order["order_amount"]), 2)
    if round(float(refund_inr), 2) != expected_refund:
        raise ValueError("Refund amount must equal the completed order amount")
    if escalate and expected_refund <= 1000:
        raise ValueError("Only refunds above INR 1000 are escalated")
    if not escalate and expected_refund > 1000:
        raise ValueError("Refunds above INR 1000 require human escalation")
    return str(order["order_id"])


def issue_refund(order_id: str, refund_inr: float) -> dict[str, float]:
    canonical_order_id = _validate_refund_action(
        order_id, refund_inr, escalate=False
    )
    return _write_ledger("issue_refund", canonical_order_id, refund_inr)


def escalate_to_human(order_id: str, refund_inr: float) -> dict[str, float]:
    canonical_order_id = _validate_refund_action(
        order_id, refund_inr, escalate=True
    )
    return _write_ledger("escalate_to_human", canonical_order_id, refund_inr)


TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Look up an order and return its status, amount, and delivery time.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "search_policy",
        "description": "Retrieve the three policy chunks most relevant to a query.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "calculate_refund",
        "description": "Calculate the late-delivery refund amount for a completed order.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": ["number", "null"]},
            },
            "required": ["order_amount", "delivery_minutes"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Record an automatically approved refund in the refund ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"},
            },
            "required": ["order_id", "refund_inr"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Record a refund above INR 1000 for human approval.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"},
            },
            "required": ["order_id", "refund_inr"],
        },
    },
]

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}

AGENT_SYSTEM_PROMPT = (
    "AGENT: Resolve simulated refund requests using the tools provided in the "
    "conversation. Verify order status and the refund policy before calculating "
    "or issuing a refund. Orders over INR 1000 require human escalation."
)


class SimulatedAgentCrash(RuntimeError):
    """Raised after a requested tool result has been durably checkpointed."""


def _checkpoint_path(run_id: str) -> Path:
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a filename-safe identifier")
    return STATE_DIR / f"{run_id}.json"


def _save_checkpoint(checkpoint: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(checkpoint["run_id"])
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _fallback(status: str) -> str:
    return f"{status}: Please hand this request to a human support agent for review."


def _status_from_answer(final_answer: str) -> str:
    return final_answer.split(":", 1)[0].strip()


def _agent_user_content(user_message: str) -> str:
    """Expose a lookup alias because the supplied mock parses six-digit IDs only."""
    import re

    def padded(match: Any) -> str:
        digits = match.group(1)
        return f"ORD-{digits.zfill(6)}" if len(digits) < 6 else match.group(0)

    return re.sub(r"\bORD-(\d+)\b", padded, user_message)


def _tool_call_fields(tool_call: Any) -> tuple[str, str, dict[str, Any]]:
    call_id = tool_call.id
    name = tool_call.name
    arguments = tool_call.arguments
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise ValueError(f"Tool arguments for {name} must be a JSON object")
    return call_id, name, arguments


def _agent_loop(
    checkpoint: dict[str, Any],
    max_iterations: int,
    token_limit: int,
    time_limit_seconds: float,
    crash_after_tool_execution: int | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    initial_tool_count = len(checkpoint["tools_executed"])
    while checkpoint["llm_calls"] < max_iterations:
        if time.monotonic() - started >= time_limit_seconds:
            checkpoint["status"] = "FALLBACK_TIMEOUT"
            checkpoint["final_answer"] = _fallback(checkpoint["status"])
            break
        response = mock_llm.chat(checkpoint["messages"], tools=TOOLS_SPEC)
        checkpoint["llm_calls"] += 1
        checkpoint["tokens_used"] += response.usage.total_tokens
        if checkpoint["tokens_used"] > token_limit:
            checkpoint["status"] = "FALLBACK_TOKEN_LIMIT"
            checkpoint["final_answer"] = _fallback(checkpoint["status"])
            break

        reply = response.choices[0].message
        tool_calls = reply.tool_calls or []
        if not tool_calls:
            checkpoint["final_answer"] = reply.content or ""
            checkpoint["status"] = _status_from_answer(checkpoint["final_answer"])
            break

        assistant_calls = []
        for tool_call in tool_calls:
            call_id, name, arguments = _tool_call_fields(tool_call)
            assistant_calls.append(
                {"id": call_id, "name": name, "arguments": arguments}
            )
        checkpoint["messages"].append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": assistant_calls,
            }
        )

        for call_id, name, arguments in (
            _tool_call_fields(tool_call) for tool_call in tool_calls
        ):
            function = TOOL_MAP.get(name)
            if function is None:
                observation: dict[str, Any] = {"error": f"unknown tool: {name}"}
            else:
                observation = function(**arguments)
            checkpoint["steps"].append(
                {"tool_name": name, "arguments": arguments, "observation": observation}
            )
            checkpoint["tools_executed"].append(name)
            checkpoint["messages"].append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": json.dumps(observation, ensure_ascii=False),
                }
            )
            checkpoint["step_number"] += 1
            _save_checkpoint(checkpoint)
            if (
                crash_after_tool_execution is not None
                and len(checkpoint["tools_executed"]) - initial_tool_count
                >= crash_after_tool_execution
            ):
                raise SimulatedAgentCrash(
                    f"Simulated crash after {crash_after_tool_execution} tool executions"
                )
    else:
        checkpoint["status"] = "FALLBACK_MAX_ITERATIONS"
        checkpoint["final_answer"] = _fallback(checkpoint["status"])

    checkpoint.setdefault("status", "FALLBACK_MAX_ITERATIONS")
    checkpoint.setdefault("final_answer", _fallback(checkpoint["status"]))
    _save_checkpoint(checkpoint)
    return {
        "run_id": checkpoint["run_id"],
        "scenario_id": checkpoint["scenario_id"],
        "status": checkpoint["status"],
        "final_answer": checkpoint["final_answer"],
        "steps": checkpoint["steps"],
        "llm_calls": checkpoint["llm_calls"],
        "tokens_used": checkpoint["tokens_used"],
    }


def run_agent(
    user_message: str,
    scenario_id: str,
    max_iterations: int = 6,
    token_limit: int = 6000,
    time_limit_seconds: float = 10,
    *,
    run_id: str | None = None,
    crash_after_tool_execution: int | None = None,
) -> dict[str, Any]:
    actual_run_id = run_id or f"{scenario_id}_{uuid.uuid4().hex}"
    checkpoint: dict[str, Any] = {
        "run_id": actual_run_id,
        "scenario_id": scenario_id,
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": _agent_user_content(user_message)},
        ],
        "tokens_used": 0,
        "step_number": 0,
        "tools_executed": [],
        "steps": [],
        "llm_calls": 0,
    }
    _save_checkpoint(checkpoint)
    return _agent_loop(
        checkpoint,
        max_iterations,
        token_limit,
        time_limit_seconds,
        crash_after_tool_execution,
    )


def resume_agent(
    run_id: str,
    max_iterations: int = 6,
    token_limit: int = 6000,
    time_limit_seconds: float = 10,
) -> dict[str, Any]:
    checkpoint = json.loads(_checkpoint_path(run_id).read_text(encoding="utf-8"))
    return _agent_loop(checkpoint, max_iterations, token_limit, time_limit_seconds)


def run_scenarios() -> list[dict[str, Any]]:
    with SCENARIOS_PATH.open(encoding="utf-8-sig", newline="") as source:
        scenarios = list(csv.DictReader(source))
    runs = []
    for scenario in scenarios:
        run = run_agent(scenario["message"], scenario["scenario_id"])
        runs.append(
            {
                "scenario_id": scenario["scenario_id"],
                "status": run["status"],
                "final_answer": run["final_answer"],
                "llm_calls": run["llm_calls"],
                "tokens_used": run["tokens_used"],
                "steps": run["steps"],
            }
        )
    with AGENT_RUNS_PATH.open("w", encoding="utf-8") as destination:
        for run in runs:
            destination.write(json.dumps(run, ensure_ascii=False) + "\n")
    return runs


def run_resume_demo() -> tuple[list[str], dict[str, Any]]:
    with SCENARIOS_PATH.open(encoding="utf-8-sig", newline="") as source:
        scenario = next(row for row in csv.DictReader(source) if row["scenario_id"] == "S1")
    run_id = "crash_resume_S1"
    try:
        run_agent(
            scenario["message"],
            scenario["scenario_id"],
            run_id=run_id,
            crash_after_tool_execution=2,
        )
    except SimulatedAgentCrash:
        pass
    else:
        raise AssertionError("Expected the simulated crash after the second tool")
    checkpoint = json.loads(_checkpoint_path(run_id).read_text(encoding="utf-8"))
    tools_before_resume = list(checkpoint["tools_executed"])
    resumed = resume_agent(run_id)
    return tools_before_resume + [
        step["tool_name"] for step in resumed["steps"][len(tools_before_resume) :]
    ], resumed


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if LEDGER_PATH.exists():
        LEDGER_PATH.unlink()

    assert len(CHUNKS) == 15
    counts: dict[str, int] = {}
    for chunk in CHUNKS:
        counts[chunk["source"]] = counts.get(chunk["source"], 0) + 1
    print("Chunk counts:", dict(sorted(counts.items())))

    policy_results = evaluate_policy()
    no_overlap_results = evaluate_policy(overlap=0)
    print("Policy evaluation (overlap=10):", policy_results)
    print("Policy evaluation (overlap=0):", no_overlap_results)

    runs = run_scenarios()
    with SCENARIOS_PATH.open(encoding="utf-8-sig", newline="") as source:
        scenarios = {row["scenario_id"]: row for row in csv.DictReader(source)}
    for run in runs:
        expected = scenarios[run["scenario_id"]]
        if expected["expected_outcome"] == "FALLBACK":
            assert run["status"] == "FALLBACK_MAX_ITERATIONS"
        else:
            assert run["status"] == expected["expected_outcome"]
        refund_steps = [
            step
            for step in run["steps"]
            if step["tool_name"] == "calculate_refund"
        ]
        actual_refund = (
            refund_steps[-1]["observation"]["refund_inr"] if refund_steps else 0.0
        )
        assert abs(actual_refund - float(expected["expected_refund_inr"])) < 0.01
        print(
            f"{run['scenario_id']}: {run['status']} | {run['final_answer']} | "
            f"calls={run['llm_calls']} tokens={run['tokens_used']}"
        )

    token_demo = run_agent(
        "My order ORD-1000 reached me very late. I would like a refund.",
        "S1_token_limit_200",
        token_limit=200,
    )
    timeout_demo = run_agent(
        "My order ORD-1000 reached me very late. I would like a refund.",
        "S1_timeout_zero",
        time_limit_seconds=0,
    )
    assert token_demo["status"] == "FALLBACK_TOKEN_LIMIT"
    assert timeout_demo["status"] == "FALLBACK_TIMEOUT"
    tools, resumed = run_resume_demo()
    assert tools == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert resumed["status"] == "REFUND"
    assert resumed["steps"][-1]["observation"]["refund_inr"] == float(
        scenarios["S1"]["expected_refund_inr"]
    )
    print("Token-limit demonstration:", token_demo["status"])
    print("Timeout demonstration:", timeout_demo["status"])
    print("Crash/resume tools:", tools, "final:", resumed["status"], resumed["final_answer"])


if __name__ == "__main__":
    main()
