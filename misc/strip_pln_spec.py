"""
One-shot utility: strip the legacy `pln_spec` key from saved DSPy
program demos.

Background
----------
Before the nl2pln.py refactor that moved `pln_spec` from a per-call
InputField into the signature instruction, every saved demo carried a
full snapshot of `pln_spec` in its dict.  For programs trained against
chainer_analysis.txt, that's ~28KB per demo — adding up to ~173KB of
duplicated content per LM call for a 6-demo program.

After the refactor, the runtime adapter no longer renders this field
(the signature doesn't list it as an InputField anymore), so the bloat
is already eliminated at *prompt-rendering* time on load.  But the
saved JSON files on disk still carry the legacy data until each program
is loaded-and-re-saved.

This script does that bulk-cleanup at-rest: it loads each JSON,
removes any `pln_spec` key from each demo, and rewrites the file.
Optional --dry-run mode reports the changes without writing.

The instruction strings (signature.instructions) are untouched —
they don't carry pln_spec content for legacy programs (it was a
separate field).  Load-time injection adds pln_spec fresh from the
current `nl2pln.pln_spec`, so it works regardless of what the saved
instruction looks like.

Usage
-----
    # Clean every program in the default directory
    python misc/strip_pln_spec.py

    # Clean specific files
    python misc/strip_pln_spec.py programs/simba_all_roman.json programs/bfs_instruct_analysis.json

    # Dry-run (report what would change, don't write)
    python misc/strip_pln_spec.py --dry-run

    # Use a different programs directory
    python misc/strip_pln_spec.py --dir other_programs/

Safe to re-run.  Files without legacy `pln_spec` keys are left untouched.
Existing files are overwritten in place; no backup is created — git
history is the recommended safety net (commit before running if you
want a reversible state).
"""
import argparse
import json
import pathlib
import sys


def strip_program(path: pathlib.Path, dry_run: bool = False) -> tuple[int, int, int]:
    """Strip pln_spec from one program JSON.  Returns
    (demos_touched, bytes_before, bytes_after).  If dry_run, no write.
    """
    raw = path.read_text(encoding="utf-8")
    bytes_before = len(raw.encode("utf-8"))
    data = json.loads(raw)

    demos_touched = 0
    for key, val in data.items():
        # Predictor blocks are top-level dicts containing a "demos" list.
        # Anything else (e.g. "metadata") is left untouched.
        if isinstance(val, dict) and isinstance(val.get("demos"), list):
            for demo in val["demos"]:
                if isinstance(demo, dict) and "pln_spec" in demo:
                    del demo["pln_spec"]
                    demos_touched += 1

    if demos_touched == 0:
        return 0, bytes_before, bytes_before

    new_raw = json.dumps(data, indent=2, ensure_ascii=False)
    bytes_after = len(new_raw.encode("utf-8"))

    if not dry_run:
        path.write_text(new_raw, encoding="utf-8")

    return demos_touched, bytes_before, bytes_after


def main():
    p = argparse.ArgumentParser(
        description="Bulk-strip legacy pln_spec from saved DSPy program demos.",
    )
    p.add_argument(
        "paths",
        nargs="*",
        help="Specific program JSON paths.  If omitted, processes every "
             "*.json in --dir (default: programs/).",
    )
    p.add_argument(
        "--dir", default="programs",
        help="Directory to scan when no paths are given (default: programs/).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Report what would change without writing.",
    )
    args = p.parse_args()

    if args.paths:
        targets = [pathlib.Path(p) for p in args.paths]
    else:
        targets = sorted(pathlib.Path(args.dir).glob("*.json"))

    if not targets:
        print(f"No JSON files found.")
        sys.exit(1)

    total_demos_touched = 0
    total_bytes_saved = 0
    files_changed = 0

    print(f"{'file':<60} {'demos':>7} {'before':>12} {'after':>12} {'saved':>10}")
    print("-" * 110)

    for path in targets:
        try:
            demos_touched, before, after = strip_program(path, dry_run=args.dry_run)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"  SKIP {path}: {type(e).__name__}: {e}")
            continue
        saved = before - after
        if demos_touched > 0:
            files_changed += 1
            total_demos_touched += demos_touched
            total_bytes_saved += saved
            marker = "(dry-run)" if args.dry_run else ""
            print(f"{str(path):<60} {demos_touched:>7} {before:>12,} {after:>12,} {saved:>10,} {marker}")
        else:
            print(f"{str(path):<60} {'-':>7} {before:>12,} {'-':>12} {'-':>10}  no change")

    print("-" * 110)
    verb = "would clean" if args.dry_run else "cleaned"
    print(
        f"Summary: {verb} {total_demos_touched} demos across {files_changed} "
        f"files; {total_bytes_saved:,} bytes saved on disk."
    )


if __name__ == "__main__":
    main()
