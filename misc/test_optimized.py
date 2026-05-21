"""
Minimal script to test the SIMBA-optimized NL2PLN program.

Loads programs/simba_all.json into NL2PLNModule's underlying ChainOfThought
predictor and translates a few hardcoded sentences + a question into PLN.

Run from anywhere (paths resolve relative to this file's location):
    python misc/test_optimized.py
Edit SENTENCES / QUESTION below to test different inputs.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import dspy
from nl2pln import NL2PLNModule, pln_spec

OPTIMIZED_PROGRAM = REPO_ROOT / "programs" / "simba_all.json"

# === Edit these to test different inputs =====================================
SENTENCES = [
    "Fido is a dog.",
    "Dogs are animals.",
]
QUESTION = "What is Fido?"
# =============================================================================


def main():
    dspy.configure(lm=dspy.LM("openai/gpt-5.4-mini", reasoning_effort="high"))

    module = NL2PLNModule()
    module.load(str(OPTIMIZED_PROGRAM))
    predictor = module.nl2pln  # the inner ChainOfThought

    # Pass 1: declarative sentences -> KB statements
    print("=== Translating declarative sentences ===")
    print(f"  Input: {SENTENCES}")
    result = predictor(sentences=SENTENCES, context=[], pln_spec=pln_spec)

    if getattr(result, "reasoning", None):
        print(f"\n  Reasoning:\n{result.reasoning}")

    print("\n  PLN statements:")
    for s in (result.statements or []):
        print(f"    {s}")

    if result.queries:
        print("\n  PLN queries (unexpected for pure declaratives):")
        for q in result.queries:
            print(f"    {q}")

    # Pass 2: question -> PLN query, with the just-produced statements as context
    print(f"\n=== Translating question ===")
    print(f"  Input: {QUESTION!r}")
    context_for_question = list(result.statements or [])
    result_q = predictor(
        sentences=[QUESTION],
        context=context_for_question,
        pln_spec=pln_spec,
    )

    if getattr(result_q, "reasoning", None):
        print(f"\n  Reasoning:\n{result_q.reasoning}")

    print("\n  PLN queries:")
    for q in (result_q.queries or []):
        print(f"    {q}")

    if result_q.statements:
        print("\n  Additional statements (often empty for a question):")
        for s in result_q.statements:
            print(f"    {s}")


if __name__ == "__main__":
    main()
