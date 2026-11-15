"""Eval dataset -> Langfuse dataset -> experiment run with scores.

Run: python -m evals.eval   (results show under Langfuse > Datasets > acme-support-eval)
"""

from datetime import datetime

from langfuse import Evaluation, Langfuse
from langfuse.langchain import CallbackHandler

from app.bootstrap import build_agent, run
from app.config import get_settings

DATASET = "acme-support-eval"
ITEMS = [
    {
        "id": "refund-in-window",
        "input": "Can I get a refund for order A-1001?",
        "expected": {"keywords": ["full refund", "14 days"], "tools": ["get_order_status", "search_knowledge_base"]},
    },
    {
        "id": "refund-prorated",
        "input": "I want my money back for order A-1002.",
        "expected": {"keywords": ["prorated"], "tools": ["get_order_status", "search_knowledge_base"]},
    },
    {
        "id": "sla",
        "input": "Does the Pro plan have an uptime SLA?",
        "expected": {"keywords": ["best-effort"], "tools": ["search_knowledge_base"]},
    },
    {
        "id": "pricing",
        "input": "How much is the Team plan?",
        "expected": {"keywords": ["$50"], "tools": ["search_knowledge_base"]},
    },
    {
        "id": "off-topic-geo",
        "input": "What is the capital of France?",
        "expected": {"keywords": ["can only help"], "tools": []},
    },
    {
        "id": "off-topic-creative",
        "input": "Write me a short poem about the ocean.",
        "expected": {"keywords": ["can only help"], "tools": []},
    },
    {
        "id": "mixed",
        "input": "Can I get a refund for A-1001, and what's the capital of Spain?",
        "expected": {
            "keywords": ["full refund", "can only help"],
            "tools": ["get_order_status", "search_knowledge_base"],
        },
    },
]


def seed(lf: Langfuse) -> None:
    lf.create_dataset(name=DATASET)
    for it in ITEMS:  # fixed id => idempotent upsert
        lf.create_dataset_item(
            dataset_name=DATASET,
            id=it["id"],
            input={"question": it["input"]},
            expected_output=it["expected"],
        )


def score(output: dict, expected: dict) -> dict[str, float]:
    answer = output["answer"].lower()
    kw = sum(k.lower() in answer for k in expected["keywords"]) / len(expected["keywords"])
    tools = float(set(expected["tools"]) <= set(output["tools_used"]))
    return {"keyword_recall": kw, "tool_correctness": tools}


def evaluator(*, output, expected_output, **kwargs) -> list[Evaluation]:
    return [Evaluation(name=name, value=value) for name, value in score(output, expected_output).items()]


def main() -> None:
    settings = get_settings()
    lf = Langfuse(
        public_key=settings.langfuse_public_key.get_secret_value(),
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        host=settings.langfuse_host,
    )
    graph = build_agent(settings)

    def task(*, item, **kwargs):
        return run(graph, item.input["question"], callbacks=[CallbackHandler()])

    seed(lf)

    result = lf.run_experiment(
        name=DATASET,
        run_name=f"{settings.chat_model}-{datetime.now():%Y%m%d-%H%M}",
        data=lf.get_dataset(DATASET).items,
        task=task,
        evaluators=[evaluator],
    )
    print(result.format(include_item_results=True))
    lf.flush()


if __name__ == "__main__":
    main()
