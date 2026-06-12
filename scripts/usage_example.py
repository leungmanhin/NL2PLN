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
    p.add_argument("--pln-spec-file", default=None,
                   help="pln_spec file (e.g. bootstrap/chainer_analysis.txt).  If "
                        "omitted and --chainer is 'pettachainer', uses its bundled "
                        "LLM_RULE_SPEC.md via get_language_spec.")
    from chainers import add_chainer_arg
    add_chainer_arg(p)
    return p.parse_args()


def main():
    args = parse_args()

    import dspy
    import nl2pln
    from nl2pln import NL2PLNModule
    from chainers import configure_chainer

    # Configure the LM the program was optimized against.
    lm_kwargs = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else {}
    lm_kwargs["max_tokens"] = args.max_tokens
    dspy.configure(lm=dspy.LM(args.model, **lm_kwargs))

    # Select the chainer to run the generated PLN against (no default).
    configure_chainer(args.chainer)

    # Set the pln_spec the program expects.  Priority: explicit --pln-spec-file;
    # else, for PeTTaChainer only, its bundled LLM_RULE_SPEC.md (what the bundled
    # simba_all.json was optimized against).  For other chainers, pass the
    # matching spec file (e.g. bootstrap/chainer_analysis_libpln.txt).
    if args.pln_spec_file:
        nl2pln.pln_spec = pathlib.Path(args.pln_spec_file).read_text(encoding="utf-8")
        print(f"  pln_spec: {args.pln_spec_file}")
    elif args.chainer == "pettachainer":
        from pettachainer import get_language_spec
        nl2pln.pln_spec = get_language_spec(llm_focused=True)
        print("  pln_spec: PeTTaChainer LLM_RULE_SPEC.md (get_language_spec)")
    else:
        print("  WARNING: no --pln-spec-file given and --chainer is not "
              "pettachainer; running with empty pln_spec.")

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
    chainer = nl2pln.make_chainer()
    for stmt in pred.statements:
        chainer.add_atom(stmt)
    for q, qr_list in zip(queries, pred.queries):
        print(f"Question: {q['question']}")
        for qr in qr_list:
            print(f"  Query: {qr}")
            print(f"  Proof: {chainer.query(qr)}")


if __name__ == "__main__":
    main()
