"""Eval dataset -> Langfuse dataset -> experiment run with scores.

Run: python eval.py   (results show under Langfuse > Datasets > acme-support-eval)
"""
from datetime import datetime

from langfuse import get_client
from langfuse.langchain import CallbackHandler

from agent import run

DATASET = "acme-support-eval"
ITEMS = [
    {"id": "refund-in-window", "input": "Can I get a refund for order A-1001?",
     "expected": {"keywords": ["full refund", "14 days"], "tools": ["get_order_status", "search_knowledge_base"]}},
    {"id": "refund-prorated", "input": "I want my money back for order A-1002.",
     "expected": {"keywords": ["prorated"], "tools": ["get_order_status", "search_knowledge_base"]}},
    {"id": "sla", "input": "Does the Pro plan have an uptime SLA?",
     "expected": {"keywords": ["best-effort"], "tools": ["search_knowledge_base"]}},
    {"id": "pricing", "input": "How much is the Team plan?",
     "expected": {"keywords": ["$50"], "tools": ["search_knowledge_base"]}},
]

lf = get_client()


def seed() -> None:
    lf.create_dataset(name=DATASET)
    for it in ITEMS:  # fixed id => idempotent upsert
        lf.create_dataset_item(dataset_name=DATASET, id=it["id"],
                               input={"question": it["input"]}, expected_output=it["expected"])


def score(output: dict, expected: dict) -> dict[str, float]:
    answer = output["answer"].lower()
    kw = sum(k.lower() in answer for k in expected["keywords"]) / len(expected["keywords"])
    tools = float(set(expected["tools"]) <= set(output["tools_used"]))
    return {"keyword_recall": kw, "tool_correctness": tools}


def main() -> None:
    seed()
    run_name = f"gpt-4o-mini-{datetime.now():%Y%m%d-%H%M}"
    totals: dict[str, float] = {}
    for item in lf.get_dataset(DATASET).items:
        with item.run(run_name=run_name) as span:  # links the trace to the dataset item
            out = run(item.input["question"], callbacks=[CallbackHandler()])
            span.update_trace(input=item.input, output=out)
            for name, value in score(out, item.expected_output).items():
                span.score_trace(name=name, value=value)
                totals[name] = totals.get(name, 0) + value
            print(f"{item.id:<18} tools={out['tools_used']}")
    lf.flush()
    n = len(ITEMS)
    print({k: round(v / n, 2) for k, v in totals.items()})


if __name__ == "__main__":
    main()
