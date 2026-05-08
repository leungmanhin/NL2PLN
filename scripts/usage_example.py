"""
End-to-end deployment example for an optimized NL2PLN program.

Loads `programs/simba_all.json` (the bundled optimized program), runs
single-puzzle inference on a hand-coded example, and feeds the resulting
PLN statements/queries to PeTTaChainer to produce proofs.

This is a deployment-side demo — to evaluate a program against a dataset,
use scripts/eval_program.py instead.

Usage:
    python scripts/usage_example.py
    python scripts/usage_example.py --program programs/some_other.json
    python scripts/usage_example.py --model openai/gpt-5.4-mini
"""
import argparse
import pathlib
import sys

# Allow `from nl2pln import ...` even when run from project root.
_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def parse_args():
    p = argparse.ArgumentParser(
        description="Run a single-puzzle inference with a saved NL2PLN program"
    )
    p.add_argument("--program", default="programs/simba_all.json",
                   help="Path to the optimized program JSON")
    p.add_argument("--model", default="openai/gpt-5.4-mini",
                   help="LiteLLM model id for the task LM "
                        "(should match the LM the program was optimized against)")
    p.add_argument("--reasoning-effort", default="high",
                   help="Reasoning effort for reasoning-capable models: "
                        "none, low, medium, high, xhigh")
    p.add_argument("--max-tokens", type=int, default=32768,
                   help="Output token cap for the task LM")
    return p.parse_args()


def main():
    args = parse_args()

    import dspy
    from pettachainer import PeTTaChainer, get_language_spec
    import nl2pln
    from nl2pln import NL2PLNModule

    # Configure the LM the program was optimized against.
    lm_kwargs = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else {}
    lm_kwargs["max_tokens"] = args.max_tokens
    dspy.configure(lm=dspy.LM(args.model, **lm_kwargs))

    # Set the pln_spec the bundled program was optimized against
    # (PeTTaChainer's LLM_RULE_SPEC.md, accessed via get_language_spec).
    # If you swap the program for one trained against bootstrap/chainer_analysis.txt
    # or any other spec, set nl2pln.pln_spec accordingly here.
    nl2pln.pln_spec = get_language_spec(llm_focused=True)

    # Hand-coded single-puzzle example.
    sentences = [
        "Fido is a dog.",
        "Dogs are animals.",
    ]
    queries = [
        {"question": "What is Fido?", "expected_answer": "An animal"},
    ]

    # Load the optimized program.
    print(f"Loading program from {args.program} ...")
    module = NL2PLNModule()
    module.load(args.program)

    print("\n=== Input ===")
    print("Sentences:")
    for s in sentences:
        print(f"  {s}")
    print("Question:")
    for q in queries:
        print(f"  {q['question']}")

    # Run inference: NL → PLN.
    pred = module(sentences=sentences, queries=queries)

    print("\n=== Generated PLN ===")
    print("Statements (added to the chainer's KB):")
    for stmt in pred.statements:
        print(f"  {stmt}")
    print("Queries (one inner list per question; inner queries are conjunctive):")
    for q, qr_list in zip(queries, pred.queries):
        print(f"  For '{q['question']}':")
        for qr in qr_list:
            print(f"    {qr}")

    # Apply: feed statements to a chainer instance, run each query.
    print("\n=== Chainer execution ===")
    chainer = PeTTaChainer()
    for stmt in pred.statements:
        chainer.add_atom(stmt)
    for q, qr_list in zip(queries, pred.queries):
        print(f"Question: {q['question']}")
        for qr in qr_list:
            print(f"  Query: {qr}")
            print(f"  Proof: {chainer.query(qr)}")


if __name__ == "__main__":
    main()
