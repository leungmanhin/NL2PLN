"""
Generate per-phenomenon training data for NL2PLN.

Reads the phenomena list from `linguistic_phenomena.txt` (produced by
Step 2 of `bootstrap_chainer.py`) and uses an LLM to generate a batch
of diverse training examples for each phenomenon. Each example contains
1-3 sentences and 1-3 question/expected_answer pairs, matching the
schema in `data/all.json` so the output can be passed directly to
`NL2PLNModule` / SIMBA / GEPA via `--dataset`.

`data/all.json` is kept as a held-out eval set; this script produces
a *new* file (default `data/generated.json`).

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
_DEFAULT_PHENOMENA = _PROJECT_ROOT / "linguistic_phenomena.txt"
_DEFAULT_SEED_DATA = _PROJECT_ROOT / "data" / "all.json"
_DEFAULT_OUTPUT = _PROJECT_ROOT / "data" / "generated.json"


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def _parse_phenomena(path: pathlib.Path) -> list[str]:
    """
    Parse linguistic_phenomena.txt into a list of phenomenon strings.
    Each entry in the file looks like:
        1. Entity classification: <description>. Examples: <sentences>.
    Returns the per-phenomenon text (without the leading number/dot).
    """
    text = path.read_text(encoding="utf-8")
    entries = re.split(r"(?m)^\d+\.\s+", text)
    return [e.strip() for e in entries if e.strip()]


def _load_seed_examples(path: pathlib.Path, count: int) -> list[dict]:
    """Load a few examples from all.json to show the target schema."""
    if count <= 0:
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data[:count]


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

    The output schema is enforced by the response format (see the
    GeneratedExample type). Focus your effort on content quality:
    phenomenon coverage, answer derivability, and diversity.

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
      - Use different entity names across examples (varied cultural
        origin: Nadia, Carlos, Amina, Ethan, Priya, Dmitri, etc., plus
        common nouns like "the cat", "the lamp", "the teacher").
      - Use different verbs, relations, and topic domains (cooking,
        work, nature, travel, sports, family, weather, etc.).
      - Vary sentence structures and lengths.
      - Do NOT repeat a template across examples. If example 1 is
        "<Name> is a <profession>", do not make example 2 the same
        template with different names.

    This is NL-only — do NOT emit logic syntax or reference any
    formalism. Produce natural English only.
    """
    phenomenon: str = dspy.InputField(
        desc="The target linguistic phenomenon to demonstrate, including "
             "its description and example English sentences."
    )
    seed_examples: str = dspy.InputField(
        desc="A few existing examples (JSON) for STYLE and DIVERSITY "
             "reference only — the output schema is already enforced by "
             "the response format. Use these to calibrate question style "
             "and answer phrasing, and to avoid duplicating their "
             "specific entities or scenarios. May be empty."
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

    Reply with exactly "yes" if all hold, or "no: <brief reason>"
    otherwise.
    """
    sentences: list[str] = dspy.InputField(
        desc="The sentences the parser would receive."
    )
    queries: list[dict] = dspy.InputField(
        desc='List of {"question", "expected_answer"} pairs to verify.'
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
    phenomena: list[str],
    per_phenomenon: int,
    seed_examples: list[dict],
    verify: bool,
) -> list[dict]:
    """
    For each phenomenon, one batch LLM call produces `per_phenomenon`
    examples. Each accepted example is deduped against prior accepted
    ones and (optionally) passed through a second LLM call that
    validates the sentence→answer relationship.
    """
    all_examples: list[dict] = []
    seen_keys: set[tuple] = set()
    seed_json = json.dumps(seed_examples, indent=2, ensure_ascii=False)

    generator = dspy.Predict(GenerateBatchSignature)
    verifier = dspy.Predict(VerifyExampleSignature) if verify else None

    for i, phenomenon in enumerate(phenomena, 1):
        short_name = phenomenon.split(":", 1)[0][:48]
        print(f"\n[{i}/{len(phenomena)}] {short_name} ...")

        try:
            result = generator(
                phenomenon=phenomenon,
                seed_examples=seed_json,
                count=per_phenomenon,
            )
        except Exception as e:
            print(f"  ERROR generating: {e}")
            continue

        raw_batch = result.examples or []
        kept = 0
        for raw in raw_batch:
            ex = _to_example_dict(raw)
            if ex is None:
                continue
            sentences = ex.get("sentences") or []
            queries = ex.get("queries") or []
            if not sentences or not queries:
                continue  # Pydantic validation should prevent this, but defensive

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
            })
            kept += 1
            q0 = queries[0]["question"][:40]
            print(f"  OK [{kept}/{per_phenomenon}]: "
                  f"{sentences[0][:50]}... Q: {q0}...")

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
        "--phenomena",
        default=str(_DEFAULT_PHENOMENA),
        help=f"Path to linguistic_phenomena.txt (default: {_DEFAULT_PHENOMENA})",
    )
    parser.add_argument(
        "--seed-data",
        default=str(_DEFAULT_SEED_DATA),
        help=f"JSON file providing format seeds (default: {_DEFAULT_SEED_DATA})",
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
        help="Examples to generate per phenomenon (default: 5)",
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=3,
        help="Seeds shown to the generator for format reference; set to 0 "
             "to avoid any test-set contamination (default: 3)",
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

    phenomena = _parse_phenomena(pathlib.Path(args.phenomena))
    print(f"Loaded {len(phenomena)} phenomena from {args.phenomena}")

    seed_examples = _load_seed_examples(
        pathlib.Path(args.seed_data), count=args.seed_count,
    )
    print(f"Loaded {len(seed_examples)} seed examples from {args.seed_data}")

    # temperature=1.0 + cache=False are essential: without them, batches
    # would be deterministic and identical across re-runs.
    lm_kwargs = {"temperature": 1.0, "cache": False,
                 "timeout": 60, "num_retries": 3}
    if args.effort:
        lm_kwargs["reasoning_effort"] = args.effort
    lm = dspy.LM(args.model, **lm_kwargs)
    dspy.configure(lm=lm)

    examples = generate_per_phenomenon(
        phenomena=phenomena,
        per_phenomenon=args.per_phenomenon,
        seed_examples=seed_examples,
        verify=args.verify,
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
