"""
Generate per-phenomenon training data for NL2PLN.

Reads `bootstrap/phenomenon_feature_mapping.txt` (produced by Step 3 of
`bootstrap_chainer.py`) — a per-phenomenon mapping that pairs each
linguistic phenomenon with the chainer-specific features it should
exercise and constraint templates for the expected proofs.  Uses an LLM
to generate a batch of diverse training examples per phenomenon-block.
Each example contains 1-3 sentences and 1-3 question/expected_answer
pairs, optionally with chainer-feature constraints on each query.  The
output matches the schema in `data/all.json` plus the new `constraints`
field, so it flows directly into NL2PLNModule / the optimizers via
`--dataset`.

Each generated example is optionally verified by a second LLM call
that checks the answer is derivable from the sentences alone, the
example is consistent, and the answer is unambiguous.

Usage:
    python src/generate_data.py
    python src/generate_data.py --per-phenomenon 10
    python src/generate_data.py --no-verify --per-phenomenon 3
"""
import argparse
import json
import logging
import pathlib
import re

import dspy
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
_DEFAULT_MAPPING = _PROJECT_ROOT / "bootstrap" / "phenomenon_feature_mapping.txt"
_DEFAULT_OUTPUT = _PROJECT_ROOT / "data" / "generated.json"


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def _parse_mapping_entries(path: pathlib.Path) -> list[str]:
    """
    Parse bootstrap/phenomenon_feature_mapping.txt into a list of
    per-phenomenon blocks.  Each entry in the file is a numbered block:

        1. Entity classification
           Recap: ...
           Chainer features to exercise: ...
           Constraint templates: ...

        2. Frequency and habituality
           ...

    Returns the per-phenomenon block text (without the leading number/dot).
    """
    text = path.read_text(encoding="utf-8")
    entries = re.split(r"(?m)^\d+\.\s+", text)
    return [e.strip() for e in entries if e.strip()]


def _short_name(entry: str) -> str:
    """
    Extract a short phenomenon name from a mapping entry's first line.
    Handles inline ("Entity classification — Recap: ..."), colon-separated
    ("Entity classification: ..."), and bare ("Entity classification\n...") forms.
    """
    first_line = entry.split("\n", 1)[0]
    return first_line.split(" — ", 1)[0].split(":", 1)[0].strip()


# ---------------------------------------------------------------------------
# Output schema (Pydantic) and DSPy Signatures
# ---------------------------------------------------------------------------

class GeneratedQuery(BaseModel):
    question: str = Field(
        description="A single natural-English wh-question "
                    "(who/what/where/when/which/how/why) or yes/no "
                    "question. Not compound."
    )
    expected_answer: str = Field(
        description="Short phrase or sentence answering the question. "
                    "Must be DERIVABLE from the sentences alone, without "
                    "outside world knowledge."
    )
    constraints: list[str] | None = Field(
        default=None,
        description="Optional list of constraints the expected proof "
                    "should satisfy, drawn from the input's constraint "
                    "templates and tailored to this puzzle (e.g. "
                    "concrete numeric ranges or operator names filled "
                    "in from the templates).  Leave null/empty when the "
                    "phenomenon-feature block lists no templates, or "
                    "when this specific puzzle doesn't call for a "
                    "constraint."
    )


class GeneratedExample(BaseModel):
    sentences: list[str] = Field(
        description="1-3 English sentences establishing the facts that "
                    "the queries will probe."
    )
    queries: list[GeneratedQuery] = Field(
        description="1-3 queries about the same sentences. Multiple "
                    "related queries on shared sentences are preferred "
                    "over single-question examples because they exercise "
                    "predicate reuse across queries."
    )


class GenerateBatchSignature(dspy.Signature):
    """
    Generate a BATCH of diverse training examples for a semantic parser.
    All examples in the batch demonstrate the SAME target linguistic
    phenomenon but use different entities, relations, topics, and
    sentence structures.

    The input `phenomenon_feature_block` is a per-phenomenon mapping
    block from bootstrap step 3.  It contains: the phenomenon name
    and recap, the chainer-specific features that phenomenon should
    exercise (if any), and constraint templates for the expected
    proofs (if any).

    The output schema is enforced by the response format (see the
    GeneratedExample type). Focus your effort on content quality:
    phenomenon coverage, answer derivability, diversity, and where
    applicable, chainer-feature exercise via the `constraints` field.

    Question style:
      - Prefer simple wh-questions (who/what/where/when/which/how/why)
        OR yes/no questions when the phenomenon naturally calls for
        them (e.g., modality: "Can X do Y?").
      - Avoid compound or multi-part questions.
      - The expected_answer must be DERIVABLE from the sentences alone,
        without outside world knowledge or commonsense inference that
        goes beyond the stated facts.

    Multiple queries per example (pressure toward atomic decomposition):
      - Prefer 2-3 related queries that share the same sentences, each
        probing the scenario from a DIFFERENT angle. This exercises
        predicate reuse AND creates metric pressure against monolithic
        representations — a single compound predicate cannot answer
        multiple specific questions about its parts.
      - For QUANTITATIVE or RELATIONAL phenomena (quantification,
        comparison, counting, measurement, degree, spatial/temporal
        relations, probability), design query sets that jointly require
        the parser to represent any numeric value, threshold, or
        compared quantity as a first-class argument of a predicate
        rather than baked into the predicate name. Some queries in the
        set should probe the value directly; others should test it
        against bounds or thresholds. A parser that emits one compound
        predicate will fail a subset of the queries; a parser that
        decomposes into a value-bearing predicate plus comparison
        primitives will answer all of them.
      - Each query must still be independently answerable from the
        sentences alone, without outside world knowledge.

    Diversity requirements for the batch:
      - Use varied entity names across examples; do not reuse the same
        characters across the batch.  Mix in common nouns and definite
        descriptions where natural rather than always using proper names.
      - Use different verbs, relations, and topic domains (cooking,
        work, nature, travel, sports, family, weather, etc.).
      - Vary sentence structures and lengths.
      - Do NOT repeat a template across examples. If example 1 is
        "<Name> is a <profession>", do not make example 2 the same
        template with different names.

    Chainer-feature constraints (when listed in the input):
      - The input may include "Chainer features to exercise" and
        "Constraint templates" sections.  When present, populate each
        query's `constraints` field with concrete constraint strings
        drawn from the templates, tailored to the specific puzzle
        (e.g. fill in concrete numeric ranges, predicate names, etc.).
      - A constraint is a short, self-contained, machine-checkable
        claim about what the expected proof should look like (e.g.
        a concrete numeric range applied to a named operator or TV
        form, instantiated from the templates).  The judge will
        verify constraints against the actual proof at scoring time.
      - Leave `constraints` null/empty if the block lists no
        templates, or if a specific puzzle doesn't call for any
        constraint.  Not every puzzle needs constraints.

    Sentences and questions must use NATURAL ENGLISH only — no logic
    syntax, no formalism references.  Constraints are the only field
    that may reference chainer-specific concepts (truth values,
    operators, etc.), and only when the input templates supply them.
    """
    phenomenon_feature_block: str = dspy.InputField(
        desc="A phenomenon-feature mapping block from bootstrap step 3, "
             "containing the linguistic phenomenon description, chainer-"
             "specific features the phenomenon should exercise, and "
             "constraint templates for the expected proofs."
    )
    count: int = dspy.InputField(
        desc="How many diverse examples to generate for this phenomenon."
    )
    examples: list[GeneratedExample] = dspy.OutputField(
        desc="A list of `count` diverse GeneratedExample instances."
    )


class VerifyExampleSignature(dspy.Signature):
    """
    Decide whether a generated training example is usable for a semantic
    parser that converts sentences to logical facts and queries against
    a KB built from those facts.

    The example is VALID only if ALL of the following hold:
      1. Every expected_answer is DERIVABLE from the sentences alone,
         without outside world knowledge. (Example: "Paris is in
         France." + Q: "Where is Paris?" + A: "France" is VALID. But
         "Paris is in France." + Q: "What country has the Eiffel
         Tower?" + A: "France" is NOT valid — it requires world
         knowledge not in the sentences.)
      2. Each question is answerable by straightforward deduction from
         the facts — no commonsense inference beyond what's stated, no
         arithmetic a simple reasoner can't do, no open-ended
         interpretation.
      3. The sentences are internally consistent (no contradictions).
      4. Each expected_answer is unambiguous given the sentences (only
         one reasonable answer).
      5. Questions are natural English wh-questions or yes/no questions;
         not multi-part or compound.

    Each query dict may carry an optional `constraints` field — that's
    the data generator's hint to the judge and is not part of what you
    verify here; only check the content correctness above.

    Reply with exactly "yes" if all hold, or "no: <brief reason>"
    otherwise.
    """
    sentences: list[str] = dspy.InputField(
        desc="The sentences the parser would receive."
    )
    queries: list[dict] = dspy.InputField(
        desc='List of query dicts.  Each has "question" and '
             '"expected_answer" (and optionally "constraints", which '
             'you may ignore for verification).'
    )
    verdict: str = dspy.OutputField(
        desc='Exactly "yes" if the example is valid, or "no: <reason>" '
             'otherwise.'
    )


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------

def _to_example_dict(raw) -> dict | None:
    """
    Coerce a batch element into a plain dict matching the target schema.
    DSPy may deserialize Pydantic-typed output fields into model instances
    or dicts depending on version; handle both.
    """
    if hasattr(raw, "model_dump"):
        return raw.model_dump()
    if isinstance(raw, dict):
        return raw
    return None


def generate_per_phenomenon(
    entries: list[str],
    per_phenomenon: int,
    verify: bool,
    max_attempts_per_phenomenon: int = 10,
) -> list[dict]:
    """
    For each phenomenon-feature block, repeatedly call the generator
    until exactly ``per_phenomenon`` *verified* examples accumulate, or
    ``max_attempts_per_phenomenon`` LLM batch calls are exhausted —
    whichever comes first.

    Each attempt requests ``max(need, 1) * 2`` examples (overshoot to
    amortize the per-call overhead so most phenomena finish in 1-2
    attempts even when the verifier rejects half).  Accepted examples
    carry a ``phenomenon`` field so downstream tooling can do
    stratified train/val splits or stratified mini-batching.

    If a phenomenon under-fills (some scope / ellipsis / quantifier-
    interaction phenomena are inherently hard for the generator to make
    verifiable), the function logs a WARN and moves on with however
    many it got.
    """
    all_examples: list[dict] = []
    seen_keys: set[tuple] = set()

    generator = dspy.Predict(GenerateBatchSignature)
    verifier = dspy.Predict(VerifyExampleSignature) if verify else None

    for i, entry in enumerate(entries, 1):
        short_name = _short_name(entry)
        print(f"\n[{i}/{len(entries)}] {short_name} ...")

        kept = 0
        attempts = 0

        while kept < per_phenomenon and attempts < max_attempts_per_phenomenon:
            attempts += 1
            need = per_phenomenon - kept
            request_count = max(need, 1) * 2

            try:
                result = generator(
                    phenomenon_feature_block=entry,
                    count=request_count,
                )
            except Exception as e:
                print(f"  ERROR generating (attempt {attempts}): {e}")
                continue

            raw_batch = result.examples or []
            for raw in raw_batch:
                if kept >= per_phenomenon:
                    break  # already have enough; don't waste verifier calls
                ex = _to_example_dict(raw)
                if ex is None:
                    continue
                sentences = ex.get("sentences") or []
                queries = ex.get("queries") or []
                if not sentences or not queries:
                    continue  # Pydantic validation should prevent; defensive

                key = (tuple(sentences), queries[0]["question"])
                if key in seen_keys:
                    print(f"  DUP: {sentences[0][:50]}... (skipped)")
                    continue
                seen_keys.add(key)

                if verifier is not None:
                    try:
                        check = verifier(sentences=sentences, queries=queries)
                        verdict = (check.verdict or "").strip().lower()
                    except Exception as e:
                        print(f"  VERIFY-ERR: {e}")
                        continue
                    if not verdict.startswith("yes"):
                        print(f"  REJECT: {sentences[0][:50]}... "
                              f"({verdict[:80]})")
                        continue

                all_examples.append({
                    "sentences": sentences,
                    "queries": queries,
                    "phenomenon": short_name,
                })
                kept += 1
                q0 = queries[0]["question"][:40]
                print(f"  OK [{kept}/{per_phenomenon}] "
                      f"(attempt {attempts}): {sentences[0][:50]}... "
                      f"Q: {q0}...")

        if kept < per_phenomenon:
            print(f"  WARN: {short_name} accepted {kept}/{per_phenomenon} "
                  f"after {attempts} attempts; moving on.")

    print(f"\nTotal accepted: {len(all_examples)}")
    return all_examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate per-phenomenon training data for NL2PLN"
    )
    parser.add_argument(
        "--mapping",
        default=str(_DEFAULT_MAPPING),
        help=f"Path to bootstrap/phenomenon_feature_mapping.txt (default: {_DEFAULT_MAPPING})",
    )
    parser.add_argument(
        "--output",
        default=str(_DEFAULT_OUTPUT),
        help=f"Output path for the generated JSON (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--per-phenomenon",
        type=int,
        default=5,
        help="Target number of VERIFIED examples to accumulate per "
             "phenomenon (default: 5).  The script retries the generator "
             "until this target is met or --max-attempts-per-phenomenon "
             "is exhausted.",
    )
    parser.add_argument(
        "--max-attempts-per-phenomenon",
        type=int,
        default=10,
        help="Cap on generator calls per phenomenon (default: 10).  "
             "Phenomena that can't yield --per-phenomenon verified "
             "examples within this many attempts get a WARN and move on.",
    )
    parser.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="Skip the LLM verification gate",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-5.4-mini",
        help="LM identifier (default: openai/gpt-5.4-mini)",
    )
    parser.add_argument(
        "--effort",
        default="low",
        help="Reasoning effort (low/medium/high/xhigh); empty string to omit",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    entries = _parse_mapping_entries(pathlib.Path(args.mapping))
    print(f"Loaded {len(entries)} phenomenon-feature blocks from {args.mapping}")

    # temperature=1.0 + cache=False are essential: without them, batches
    # would be deterministic and identical across re-runs.
    lm_kwargs = {"temperature": 1.0, "cache": False,
                 "timeout": 60, "num_retries": 3}
    if args.effort:
        lm_kwargs["reasoning_effort"] = args.effort
    lm = dspy.LM(args.model, **lm_kwargs)
    dspy.configure(lm=lm)

    examples = generate_per_phenomenon(
        entries=entries,
        per_phenomenon=args.per_phenomenon,
        verify=args.verify,
        max_attempts_per_phenomenon=args.max_attempts_per_phenomenon,
    )

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(examples, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved {len(examples)} examples to {out_path}")


if __name__ == "__main__":
    main()
