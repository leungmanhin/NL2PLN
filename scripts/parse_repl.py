"""
Interactive NL->PLN parse REPL.

Loads a saved NL2PLN program and translates whatever you type — a sentence
or a question — into PLN, printing the program's output.  It ONLY runs the
parser; it does not execute any chainer, so it's a quick way to eyeball
what a trained program produces.

The pln_spec the program was trained with is recovered from the saved
program file automatically (NL2PLNModule.load re-injects nl2pln.pln_spec
into the signature), so parses match the program's trained behavior without
an extra flag.

Usage:
    python scripts/parse_repl.py --program programs/simba_gpt_generated_sig_analysis_neodav.json
    python scripts/parse_repl.py --program <path> --model openai/gpt-5.4-mini --reasoning-effort high

Type a sentence or question and press Enter.  Ctrl-D (or 'quit'/'exit') stops.
"""
import argparse
import json
import pathlib
import sys

# Allow `from nl2pln import ...` even when run from the project root.
_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

_PLN_SPEC_BEGIN = "<!-- BEGIN_PLN_SPEC -->"
_PLN_SPEC_END = "<!-- END_PLN_SPEC -->"


def parse_args():
    p = argparse.ArgumentParser(
        description="Interactive NL->PLN parse REPL for a saved NL2PLN program"
    )
    p.add_argument("--program", required=True,
                   help="Path to the optimized program JSON")
    p.add_argument("--model", default="openai/gpt-5.4-mini",
                   help="LiteLLM model id for the task LM "
                        "(should match the LM the program was optimized against)")
    p.add_argument("--reasoning-effort", default="high",
                   help="Reasoning effort for reasoning-capable models: "
                        "none, low, medium, high, xhigh")
    return p.parse_args()


def _recover_pln_spec(program_path: str) -> str:
    """Recover the pln_spec the program was saved with, from the marked
    section in its saved signature instructions.  Returns "" if not found."""
    try:
        data = json.loads(pathlib.Path(program_path).read_text(encoding="utf-8"))
    except Exception:
        return ""
    found: list[str] = []

    def walk(x):
        if isinstance(x, str):
            if _PLN_SPEC_BEGIN in x and _PLN_SPEC_END in x:
                seg = x.split(_PLN_SPEC_BEGIN, 1)[1].split(_PLN_SPEC_END, 1)[0]
                seg = seg.lstrip("\n")
                if seg.startswith("## PLN spec") and "\n" in seg:
                    seg = seg.split("\n", 1)[1]
                found.append(seg.strip("\n"))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(data)
    return found[0] if found else ""


def main():
    args = parse_args()

    if not pathlib.Path(args.program).exists():
        sys.exit(f"ERROR: program not found: {args.program}")

    import dspy
    import nl2pln
    from nl2pln import NL2PLNModule

    # Configure the LM the program was optimized against.
    lm_kwargs = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else {}
    lm_kwargs["max_tokens"] = 32768   # high enough that reasoning models don't truncate
    dspy.configure(lm=dspy.LM(args.model, **lm_kwargs))

    # Recover the spec the program was trained with so parses match its
    # trained behavior (load() re-injects nl2pln.pln_spec into the signature).
    nl2pln.pln_spec = _recover_pln_spec(args.program)
    spec_note = (f"{len(nl2pln.pln_spec):,} chars recovered from program"
                 if nl2pln.pln_spec else "none found (running without pln_spec)")

    module = NL2PLNModule()
    module.load(args.program)

    print(f"Loaded {args.program}")
    print(f"  model: {args.model} (effort={args.reasoning_effort}) | pln_spec: {spec_note}")
    print("Type a sentence or question; Ctrl-D (or 'quit'/'exit') to stop.\n")

    while True:
        try:
            line = input("nl> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in {"quit", "exit"}:
            break
        try:
            pred = module.nl2pln(sentences=[line], context=[])
        except Exception as e:
            print(f"  error: {type(e).__name__}: {e}\n")
            continue
        print("  Statements:")
        for s in (pred.statements or []):
            print(f"    {s}")
        print("  Queries:")
        for q in (pred.queries or []):
            print(f"    {q}")
        print()


if __name__ == "__main__":
    main()
