"""
Build a phenomenon-stratified evaluation set for NL2PLN.

Combines a stratified subsample of data/generated.json (one example per
phenomenon by default) with the full held-out data/all.json.  Each entry
is tagged with a `source` field ("stratified" or "held_out") so
downstream eval scripts can report the two numbers separately and the
held-out integrity stays auditable.

Stratification keys on the `phenomenon` field of generated.json (added
by src/generate_data.py).  A fixed seed makes the eval set reproducible
across runs and machines: re-running with the same --seed and
--per-phenomenon yields the same selection.

Variance picture: with LLM-judge sigma ~0.1 per query and ~3 queries per
example, per-example sigma ~0.06.  Default eval set (108 + 29 = 137
examples) gives SEM ~0.005 on the combined mean, enough to resolve
~0.01 score differences confidently.

Usage:
    python scripts/build_eval_set.py
    python scripts/build_eval_set.py --per-phenomenon 2
    python scripts/build_eval_set.py --output data/eval_strat.json --no-held-out
    python scripts/build_eval_set.py --seed 42
"""
import argparse
import json
import pathlib
import random
import sys
from collections import defaultdict

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent


def parse_args():
    p = argparse.ArgumentParser(
        description="Build phenomenon-stratified evaluation set for NL2PLN"
    )
    p.add_argument(
        "--generated",
        default=str(_PROJECT_ROOT / "data" / "generated.json"),
        help="Path to generated dataset (with `phenomenon` tags)",
    )
    p.add_argument(
        "--held-out",
        default=str(_PROJECT_ROOT / "data" / "all.json"),
        help="Path to held-out dataset (e.g., data/all.json)",
    )
    p.add_argument(
        "--no-held-out",
        action="store_true",
        help="Skip the held-out portion; output stratified-only set",
    )
    p.add_argument(
        "--per-phenomenon",
        type=int,
        default=1,
        help="Examples per phenomenon to draw from generated (default: 1)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=21,
        help="Random seed for reproducible sampling (default: 21)",
    )
    p.add_argument(
        "--output",
        default=str(_PROJECT_ROOT / "data" / "eval.json"),
        help="Output file path (default: data/eval.json)",
    )
    return p.parse_args()


def _load(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"ERROR: {path} not found")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        sys.exit(f"ERROR: {path} is not a JSON list")
    return data


def _stratify(examples: list[dict], per_phenomenon: int, seed: int) -> list[dict]:
    """
    Group by `phenomenon` and sample `per_phenomenon` from each group,
    preserving first-seen phenomenon order from the source file.
    """
    rng = random.Random(seed)
    by_phenomenon: dict[str, list[dict]] = defaultdict(list)
    phenomenon_order: list[str] = []
    untagged: list[dict] = []

    for ex in examples:
        ph = ex.get("phenomenon")
        if ph is None:
            untagged.append(ex)
            continue
        if ph not in by_phenomenon:
            phenomenon_order.append(ph)
        by_phenomenon[ph].append(ex)

    if untagged:
        print(
            f"  WARN: {len(untagged)} examples have no `phenomenon` tag; skipped.",
            file=sys.stderr,
        )

    sampled: list[dict] = []
    short: list[tuple[str, int]] = []
    for ph in phenomenon_order:
        group = by_phenomenon[ph]
        if len(group) <= per_phenomenon:
            sampled.extend(group)
            if len(group) < per_phenomenon:
                short.append((ph, len(group)))
        else:
            sampled.extend(rng.sample(group, per_phenomenon))

    if short:
        print(
            f"  WARN: {len(short)} phenomena have fewer than "
            f"{per_phenomenon} examples; took all available:",
            file=sys.stderr,
        )
        for ph, n in short:
            print(f"    {ph} ({n} available)", file=sys.stderr)

    print(
        f"  Stratified: {len(by_phenomenon)} phenomena × up to "
        f"{per_phenomenon} = {len(sampled)} examples"
    )
    return sampled


def main():
    args = parse_args()

    # ---- Stratified portion (from generated.json) ----
    generated_path = pathlib.Path(args.generated)
    print(f"Loading generated set from {generated_path} ...")
    generated = _load(generated_path)
    print(f"  {len(generated)} examples")
    stratified = _stratify(generated, args.per_phenomenon, args.seed)
    for ex in stratified:
        ex["source"] = "stratified"

    # ---- Held-out portion ----
    held_out: list[dict] = []
    if not args.no_held_out:
        held_out_path = pathlib.Path(args.held_out)
        print(f"\nLoading held-out set from {held_out_path} ...")
        held_out = _load(held_out_path)
        for ex in held_out:
            ex["source"] = "held_out"
        print(f"  {len(held_out)} held-out examples")

    eval_set = stratified + held_out

    # ---- Sanity stats ----
    total_queries = sum(len(ex.get("queries", [])) for ex in eval_set)
    print(f"\nEval set: {len(eval_set)} examples, {total_queries} total queries")
    print(f"  stratified: {len(stratified)} examples")
    print(f"  held_out:   {len(held_out)} examples")

    # ---- Write ----
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(eval_set, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved: {output_path} ({output_path.stat().st_size:,} bytes)")
    print(
        f"\nReproducibility: re-run with --seed {args.seed} "
        f"--per-phenomenon {args.per_phenomenon} to regenerate the same eval set."
    )
    print(
        "Downstream: filter on ex['source'] to report stratified-vs-held-out "
        "scores separately."
    )


if __name__ == "__main__":
    main()
