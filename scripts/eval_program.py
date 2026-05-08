"""
Evaluate one or more saved NL2PLN programs against a dataset.

Reports per-program score breakdown and bucket distribution so failure
modes are visible at a glance.  The buckets correspond to the
difficulty_metric paths plus wrap_program's exception handling:

  hard_zero      score == 0.0  (program LM call failed, or judge LM
                                 raised a class-name-re-raised error)
  early_return   0.0 < score < 0.1 (chainer rejected a statement or
                                    query; metric returned the small
                                    +0.001 accumulator)
  floor          score == 0.1 (metric reached the judge but the judge
                               gave 0.0; final_score = max(..., 0.1))
  above_floor    score >  0.1 (judge gave a non-zero score)

If the dataset has a `source` field on each example (as produced by
scripts/build_eval_set.py), reports stratified vs held-out scores
separately.

For cost-sensitive sanity checks, use --syntax-only to skip the judge
LM entirely.  That mode runs the parser and the chainer (add_atom +
query) but never calls the judge, dramatically reducing OpenAI quota
spend.  Useful for measuring parse-success rate before deciding whether
to spend judge quota on a full eval.

A per-program pln_spec can be set with `path:spec_path` syntax on the
positional argument.  Programs without an explicit spec fall back to
the global `--pln-spec-file` if given, or otherwise to the module's
default pln_spec (PeTTaChainer's LLM_RULE_SPEC.md).  This makes it
easy to evaluate each program against the spec it was trained with,
side by side, in one run.

Usage:
  # Compare two programs, each with its training-time pln_spec
  python scripts/eval_program.py \
      programs/simba_all_sig.json \
      programs/simba_all_instruct_analysis.json:chainer_analysis.txt \
      --dataset data/all.json
  # (simba_all_sig.json uses module-default LLM_RULE_SPEC.md)

  # Force both programs to use the same pln_spec (deployment-fair):
  python scripts/eval_program.py \
      programs/simba_all_sig.json \
      programs/simba_all_instruct_analysis.json \
      --pln-spec-file chainer_analysis.txt \
      --dataset data/all.json

  # Mix and match — global default + per-program override
  python scripts/eval_program.py \
      programs/simba_all_sig.json:LLM_RULE_SPEC.md \
      programs/simba_all_instruct_analysis.json \
      --pln-spec-file chainer_analysis.txt \
      --dataset data/all.json

  # Include the un-optimized baseline alongside saved programs
  python scripts/eval_program.py \
      programs/simba_all_instruct_analysis.json \
      --baseline --instruction-file instructions.md \
      --pln-spec-file chainer_analysis.txt \
      --dataset data/eval.json

  # Syntax-only (no judge LM) — much cheaper, surfaces parse-success rate
  python scripts/eval_program.py \
      programs/simba_all_instruct_analysis.json:chainer_analysis.txt \
      --dataset data/eval.json --syntax-only

  # Limit to first N examples for a quick smoke test
  python scripts/eval_program.py \
      programs/simba_all_instruct_analysis.json:chainer_analysis.txt \
      --dataset data/eval.json --limit 10
"""
import argparse
import json
import os
import pathlib
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

# Allow `from nl2pln import ...` even when the script is run from project root
_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# Heavy imports (dspy, nl2pln, pettachainer) are deferred until main() runs so
# that --help works in environments missing some deps.


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate and compare NL2PLN programs on a dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "program_paths", nargs="*",
        help="One or more saved program JSON files.  Each entry may be "
             "either `path` (uses the global --pln-spec-file or the module "
             "default) or `path:spec_path` (uses that specific spec file "
             "for this program only).",
    )
    p.add_argument(
        "--baseline", action="store_true",
        help="Include the unmodified NL2PLNModule (no .load()) in the comparison",
    )
    p.add_argument(
        "--instruction-file", default=None,
        help="Override signature instruction from this file (applies to baseline only; "
             "saved programs already carry their instruction in the JSON)",
    )
    p.add_argument(
        "--pln-spec-file", default=None,
        help="Default pln_spec file for programs without a per-program override "
             "(e.g. chainer_analysis.txt).  pln_spec is a module-level constant, "
             "so it's reset before each program's evaluation.  If neither this "
             "flag nor a per-program override is given, the module default "
             "(PeTTaChainer's LLM_RULE_SPEC.md) is used.",
    )
    p.add_argument(
        "--dataset", required=True,
        help="Path to evaluation dataset JSON (e.g. data/eval.json or data/all.json)",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate only the first N examples (for quick smoke tests)",
    )
    p.add_argument(
        "--model", default="openai/gpt-5.4-mini",
        help="LiteLLM model id used for both translator and judge",
    )
    p.add_argument(
        "--reasoning-effort", default="high",
        choices=["none", "low", "medium", "high", "xhigh"],
        help="Reasoning effort for reasoning-capable models",
    )
    p.add_argument(
        "--max-tokens", type=int, default=8192,
        help="Output token cap (default: 8192).  DSPy's underlying default is "
             "~4000, which is borderline for reasoning models that consume "
             "tokens for reasoning before the final answer.  Bump higher for "
             "demo-heavy programs (long input prompts can also push reasoning "
             "to use more tokens).  Symptoms of too-low cap: empty "
             "pred.statements/pred.queries → puzzles land in hard_zero/early_return.",
    )
    p.add_argument(
        "--syntax-only", action="store_true",
        help="Skip the judge LM; report parse-success rate only.  Cheaper.",
    )
    p.add_argument(
        "--num-threads", type=int, default=10,
        help="Parallel evaluations across examples",
    )
    p.add_argument(
        "--output", default=None,
        help="Save per-puzzle results to this JSON (default: don't save)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Full evaluation (with judge)
# ---------------------------------------------------------------------------

def _eval_one_full(program, example):
    """Run a single example through program + difficulty_metric, mirroring
    simba_utils.wrap_program semantics."""
    from nl2pln import difficulty_metric
    prediction = None
    error = None
    try:
        prediction = program(**example.inputs())
    except Exception as e:
        return {
            "score": 0.0,
            "error": f"program: {type(e).__name__}: {e}",
            "pred_statements": None,
            "pred_queries": None,
            "pred_reasoning": None,
        }

    # Capture the prediction's output before computing the score, so we
    # still log it even if the metric raises.
    pred_statements = list(prediction.statements) if prediction.statements is not None else None
    pred_queries = list(prediction.queries) if prediction.queries is not None else None
    pred_reasoning = getattr(prediction, "reasoning", None)

    try:
        output = difficulty_metric(example, prediction)
        if hasattr(output, "score"):
            score = float(output.score)
        elif isinstance(output, (int, float)):
            score = float(output)
        else:
            score = 0.0
    except Exception as e:
        return {
            "score": 0.0,
            "error": f"metric: {type(e).__name__}: {e}",
            "pred_statements": pred_statements,
            "pred_queries": pred_queries,
            "pred_reasoning": pred_reasoning,
        }
    return {
        "score": score,
        "error": error,
        "pred_statements": pred_statements,
        "pred_queries": pred_queries,
        "pred_reasoning": pred_reasoning,
    }


def _classify_score(score: float) -> str:
    if score == 0.0:
        return "hard_zero"
    if score < 0.1:
        return "early_return"
    if abs(score - 0.1) < 1e-9:
        return "floor"
    return "above_floor"


def evaluate_full(program, examples, num_threads):
    results: list[dict] = [None] * len(examples)  # type: ignore
    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        future_to_idx = {
            ex.submit(_eval_one_full, program, e): i for i, e in enumerate(examples)
        }
        done = 0
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            results[idx] = fut.result()
            done += 1
            print(f"\r  {done}/{len(examples)} done", end="", flush=True)
    print()
    return results


def report_full(label: str, results: list[dict], examples: list) -> dict:
    n = len(results)
    scores = [r["score"] for r in results]
    mean = sum(scores) / n if n else 0.0

    buckets: dict[str, int] = defaultdict(int)
    for s in scores:
        buckets[_classify_score(s)] += 1

    print(f"=== {label} ===")
    print(f"  Mean score: {mean:.4f} (n={n})")
    for b in ["hard_zero", "early_return", "floor", "above_floor"]:
        c = buckets[b]
        if c:
            print(f"  {b:>15}: {c:>3} ({c / n:.1%})")

    by_source: dict[str, list[float]] = defaultdict(list)
    for ex, s in zip(examples, scores):
        src = getattr(ex, "source", None)
        if src:
            by_source[src].append(s)
    if by_source:
        print("  By source:")
        for src in sorted(by_source):
            vals = by_source[src]
            print(f"    {src:>12}: {sum(vals) / len(vals):.4f} (n={len(vals)})")

    return {
        "label": label,
        "mean": mean,
        "n": n,
        "buckets": dict(buckets),
        "by_source": {
            s: {"mean": sum(v) / len(v), "n": len(v)} for s, v in by_source.items()
        },
        "per_puzzle": [
            {
                "score": r["score"],
                "error": r.get("error"),
                "source": getattr(ex, "source", None),
                "phenomenon": getattr(ex, "phenomenon", None),
                "sentences": list(getattr(ex, "sentences", None) or []),
                "queries_nl": [
                    {"question": q.get("question"), "expected_answer": q.get("expected_answer")}
                    for q in (getattr(ex, "queries", None) or [])
                ],
                "pred_statements": r.get("pred_statements"),
                "pred_queries": r.get("pred_queries"),
                "pred_reasoning": r.get("pred_reasoning"),
            }
            for r, ex in zip(results, examples)
        ],
    }


# ---------------------------------------------------------------------------
# Syntax-only evaluation (no judge LM)
# ---------------------------------------------------------------------------

def _eval_one_syntax(program, example):
    from pettachainer import PeTTaChainer

    try:
        pred = program(**example.inputs())
    except Exception as e:
        return {
            "bucket": "program_failed",
            "n_stmts": 0, "n_queries": 0,
            "error": f"{type(e).__name__}: {e}",
            "pred_statements": None,
            "pred_queries": None,
            "pred_reasoning": None,
        }

    # Capture the full prediction once so every return path below carries it.
    pred_extras = {
        "pred_statements": list(pred.statements) if pred.statements is not None else None,
        "pred_queries": list(pred.queries) if pred.queries is not None else None,
        "pred_reasoning": getattr(pred, "reasoning", None),
    }

    stmts = list(pred.statements or [])
    if not stmts:
        return {"bucket": "no_statements", "n_stmts": 0, "n_queries": 0, **pred_extras}

    chainer = PeTTaChainer()
    n_added = 0
    rejected_stmt = None
    rejected_stmt_err = None
    for stmt in stmts:
        try:
            chainer.add_atom(stmt)
            n_added += 1
        except Exception as e:
            rejected_stmt = stmt
            rejected_stmt_err = f"{type(e).__name__}: {e}"
            break
    if rejected_stmt is not None:
        return {
            "bucket": "stmt_rejected",
            "n_stmts": n_added,
            "n_queries": 0,
            "rejected_stmt": rejected_stmt,
            "error": rejected_stmt_err,
            **pred_extras,
        }

    n_queries_run = 0
    rejected_query = None
    rejected_query_err = None
    for qr in (pred.queries or []):
        if rejected_query is not None:
            break
        for q in (qr or []):
            try:
                chainer.query(q)
                n_queries_run += 1
            except Exception as e:
                rejected_query = q
                rejected_query_err = f"{type(e).__name__}: {e}"
                break
    if rejected_query is not None:
        return {
            "bucket": "query_rejected",
            "n_stmts": n_added,
            "n_queries": n_queries_run,
            "rejected_query": rejected_query,
            "error": rejected_query_err,
            **pred_extras,
        }

    return {
        "bucket": "syntax_ok",
        "n_stmts": n_added,
        "n_queries": n_queries_run,
        **pred_extras,
    }


def evaluate_syntax(program, examples, num_threads):
    results: list[dict] = [None] * len(examples)  # type: ignore
    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        future_to_idx = {
            ex.submit(_eval_one_syntax, program, e): i for i, e in enumerate(examples)
        }
        done = 0
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            results[idx] = fut.result()
            done += 1
            print(f"\r  {done}/{len(examples)} done", end="", flush=True)
    print()
    return results


def report_syntax(label: str, results: list[dict], examples: list) -> dict:
    n = len(results)
    buckets: dict[str, int] = defaultdict(int)
    for r in results:
        buckets[r["bucket"]] += 1

    print(f"=== {label} (syntax-only) ===")
    print(f"  n={n}")
    for b in ["program_failed", "no_statements", "stmt_rejected",
              "query_rejected", "syntax_ok"]:
        c = buckets[b]
        if c:
            print(f"  {b:>15}: {c:>3} ({c / n:.1%})")

    syntax_ok = buckets["syntax_ok"]
    print(f"  Parse success rate: {syntax_ok / n:.1%}")

    samples = {
        "stmt_rejected": [r for r in results if r["bucket"] == "stmt_rejected"][:3],
        "query_rejected": [r for r in results if r["bucket"] == "query_rejected"][:3],
    }
    for k, items in samples.items():
        if items:
            print(f"  Sample {k}:")
            for r in items:
                what = r.get("rejected_stmt") or r.get("rejected_query") or "?"
                err = r.get("error") or ""
                print(f"    {what[:120]}")
                print(f"      error: {err[:120]}")

    by_source: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for ex, r in zip(examples, results):
        src = getattr(ex, "source", None)
        if src:
            by_source[src][r["bucket"]] += 1
    if by_source:
        print("  By source (parse success rate):")
        for src in sorted(by_source):
            sub = by_source[src]
            tot = sum(sub.values())
            print(f"    {src:>12}: {sub.get('syntax_ok', 0) / tot:.1%} (n={tot})")

    return {
        "label": label,
        "n": n,
        "buckets": dict(buckets),
        "per_puzzle": [
            {
                **r,
                "source": getattr(ex, "source", None),
                "phenomenon": getattr(ex, "phenomenon", None),
                "sentences": list(getattr(ex, "sentences", None) or []),
                "queries_nl": [
                    {"question": q.get("question"), "expected_answer": q.get("expected_answer")}
                    for q in (getattr(ex, "queries", None) or [])
                ],
            }
            for r, ex in zip(results, examples)
        ],
    }


# ---------------------------------------------------------------------------
# Side-by-side comparison
# ---------------------------------------------------------------------------

def print_comparison_full(reports: list[dict]):
    print("\nSide-by-side comparison:")
    width = 28
    headers = [r["label"] for r in reports]
    print(f"  {'metric':<{width}}" + "".join(f"{h:>22}" for h in headers))
    print(f"  {'mean score':<{width}}" + "".join(f"{r['mean']:>22.4f}" for r in reports))
    n = reports[0]["n"]
    for b in ["hard_zero", "early_return", "floor", "above_floor"]:
        row = []
        for r in reports:
            c = r["buckets"].get(b, 0)
            row.append(f"{c}/{r['n']} ({c / r['n']:.0%})")
        print(f"  {b:<{width}}" + "".join(f"{v:>22}" for v in row))


def print_comparison_syntax(reports: list[dict]):
    print("\nSide-by-side comparison (syntax-only):")
    width = 28
    headers = [r["label"] for r in reports]
    print(f"  {'metric':<{width}}" + "".join(f"{h:>22}" for h in headers))
    for b in ["program_failed", "no_statements", "stmt_rejected",
              "query_rejected", "syntax_ok"]:
        row = []
        for r in reports:
            c = r["buckets"].get(b, 0)
            row.append(f"{c}/{r['n']} ({c / r['n']:.0%})")
        print(f"  {b:<{width}}" + "".join(f"{v:>22}" for v in row))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_program_entry(entry: str) -> tuple[str, str | None]:
    """
    Parse a positional program argument.

    `path`            -> (path, None)               # use default pln_spec
    `path:spec_path`  -> (path, spec_path)          # per-program override
    """
    if ":" in entry:
        path_part, spec_part = entry.split(":", 1)
        return path_part, spec_part
    return entry, None


def _read_spec(spec_path: str | None) -> str | None:
    if spec_path is None:
        return None
    p = pathlib.Path(spec_path)
    if not p.exists():
        sys.exit(f"ERROR: pln_spec file {p} not found")
    return p.read_text(encoding="utf-8")


def main():
    args = parse_args()
    if not args.program_paths and not args.baseline:
        sys.exit("ERROR: provide at least one program path, or --baseline")

    # Heavy imports happen here so --help works without dspy/mlflow/etc.
    import dspy
    import nl2pln
    from nl2pln import NL2PLNModule, build_examples_from_file

    # Capture the module's original pln_spec so we can reset between
    # programs that have different spec preferences.
    ORIGINAL_PLN_SPEC = nl2pln.pln_spec
    GLOBAL_DEFAULT_SPEC = _read_spec(args.pln_spec_file)
    if GLOBAL_DEFAULT_SPEC is not None:
        print(f"  Global default pln_spec: {args.pln_spec_file} "
              f"({len(GLOBAL_DEFAULT_SPEC):,} chars)")

    # Configure DSPy LM
    lm_kwargs = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else {}
    dspy.configure(lm=dspy.LM(
        args.model,
        timeout=600,
        num_retries=3,
        max_tokens=args.max_tokens,
        **lm_kwargs,
    ))
    print(f"  LM: {args.model} (effort={args.reasoning_effort}, max_tokens={args.max_tokens})")

    # Load dataset
    dataset = build_examples_from_file(args.dataset)
    if args.limit:
        dataset = dataset[: args.limit]
    print(f"  Dataset: {args.dataset} ({len(dataset)} examples)")

    # Build labelled program list with per-program pln_spec
    # Each entry: (label, NL2PLNModule, spec_string_or_None, spec_label)
    # spec_string is the override content; None means "use global default
    # or module original".  spec_label is for human-readable reporting.
    programs: list[tuple[str, NL2PLNModule, str | None, str]] = []
    if args.baseline:
        m = NL2PLNModule()
        if args.instruction_file:
            instr_path = pathlib.Path(args.instruction_file)
            if not instr_path.exists():
                sys.exit(f"ERROR: --instruction-file {instr_path} not found")
            instruction = instr_path.read_text(encoding="utf-8")
            for _, predictor in m.named_predictors():
                predictor.signature = predictor.signature.with_instructions(instruction)
            # Manual instruction override wipes the auto-injected pln_spec
            # section.  Re-inject it so the LM still sees the spec.
            m._inject_pln_spec()
            print(f"  Baseline instruction overridden from {instr_path}")
        programs.append(("baseline", m, None, args.pln_spec_file or "module-default"))
    for entry in args.program_paths:
        path_str, spec_path = _parse_program_entry(entry)
        path_obj = pathlib.Path(path_str)
        if not path_obj.exists():
            sys.exit(f"ERROR: program path {path_str} not found")
        m = NL2PLNModule()
        m.load(str(path_obj))
        spec_content = _read_spec(spec_path)
        spec_label = spec_path or args.pln_spec_file or "module-default"
        programs.append((path_obj.stem, m, spec_content, spec_label))

    # Evaluate each program (resetting pln_spec to its per-program value first)
    reports: list[dict] = []
    for label, prog, spec_content, spec_label in programs:
        # Decide effective pln_spec for this program
        if spec_content is not None:
            nl2pln.pln_spec = spec_content
        elif GLOBAL_DEFAULT_SPEC is not None:
            nl2pln.pln_spec = GLOBAL_DEFAULT_SPEC
        else:
            nl2pln.pln_spec = ORIGINAL_PLN_SPEC
        # Re-inject pln_spec into the program's signature instructions
        # under the new global value.  Necessary because each program was
        # loaded earlier (which injected based on whatever pln_spec was
        # active then); this resyncs to the per-program spec we just set.
        prog._inject_pln_spec()

        print(f"\nEvaluating {label} (pln_spec: {spec_label}) ...")
        if args.syntax_only:
            results = evaluate_syntax(prog, dataset, args.num_threads)
            report = report_syntax(label, results, dataset)
        else:
            results = evaluate_full(prog, dataset, args.num_threads)
            report = report_full(label, results, dataset)
        report["pln_spec"] = spec_label
        reports.append(report)

    # Side-by-side
    if len(reports) > 1:
        if args.syntax_only:
            print_comparison_syntax(reports)
        else:
            print_comparison_full(reports)

    # Save
    if args.output:
        out_path = pathlib.Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nSaved per-puzzle results to {out_path}")


if __name__ == "__main__":
    main()
