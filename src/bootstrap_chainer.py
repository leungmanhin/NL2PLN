"""
Bootstrap a chainer for semantic parsing.

Reads the chainer's source code and produces the artifacts the rest of
the NL-to-PLN pipeline depends on, in three steps:

  1. Analyze the chainer (RLM) — reverse-engineer syntax, operators,
     constraints from the source code.  Includes a final cheat-sheet
     section listing each primitive with its canonical form.

  2. Enumerate linguistic phenomena (Predict) — comprehensive list of
     phenomena that the semantic parser must handle.

  3. Generate phenomenon-feature mapping (Predict) — pair each
     phenomenon from step 2 with the chainer-specific features (from
     step 1) it should exercise, and constraint templates the expected
     proofs should satisfy.  Consumed by generate_data.py as its
     primary input.

Step 1 uses RLM (Recursive LM, requires Deno) to explore the chainer
source code via iterative code execution.  Steps 2 and 3 use
dspy.Predict (single LLM call each).

Each step saves its output to a file.  Use --from-step N to skip earlier
steps and load their outputs from disk (useful for iterating on later
steps without re-running expensive earlier ones).

Usage:
    python src/bootstrap_chainer.py --chainer ../PeTTaChainer
    python src/bootstrap_chainer.py --from-step 2
    python src/bootstrap_chainer.py --from-step 3
"""
import argparse
import logging
import pathlib
import sys

import dspy
import litellm
from dspy.predict.rlm import RLM

logger = logging.getLogger(__name__)

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent

# Default output paths (all relative to project root)
_DEFAULTS = {
    "analysis": _PROJECT_ROOT / "bootstrap" / "chainer_analysis.txt",
    "phenomena": _PROJECT_ROOT / "bootstrap" / "linguistic_phenomena.txt",
    "mapping": _PROJECT_ROOT / "bootstrap" / "phenomenon_feature_mapping.txt",
}


# ---------------------------------------------------------------------------
# Source collection
# ---------------------------------------------------------------------------

_SKIP_DIRS = {".git", "__pycache__", ".mypy_cache", ".pytest_cache",
              "node_modules", ".tox", ".eggs", "venv", ".venv"}

_SOURCE_EXTENSIONS = {
    ".py", ".metta", ".js", ".ts", ".java", ".scala", ".go", ".rs",
    ".cpp", ".cc", ".c", ".h", ".hpp", ".pl", ".rb", ".sh",
    ".md", ".txt", ".cfg", ".toml", ".yaml", ".yml", ".json", ".ini",
}


def _read_chainer_source_code(chainer_dir: pathlib.Path) -> str:
    parts = []
    for f in sorted(chainer_dir.rglob("*")):
        if any(p in _SKIP_DIRS or p.endswith(".egg-info") for p in f.parts):
            continue
        if not f.is_file():
            continue
        if f.suffix.lower() not in _SOURCE_EXTENSIONS:
            continue
        try:
            parts.append(f"=== {f.relative_to(chainer_dir)} ===\n{f.read_text(encoding='utf-8')}")
        except UnicodeDecodeError:
            print(f"Skipping binary file: {f.relative_to(chainer_dir)}")
    if not parts:
        raise FileNotFoundError(f"No source files found under {chainer_dir}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def _save_text(path: pathlib.Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    print(f"  Saved: {path} ({len(content):,} chars)")


def _load_text(path: pathlib.Path, name: str) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot start from this step: {name} not found at {path}. "
            f"Run earlier steps first, or provide the file."
        )
    content = path.read_text(encoding="utf-8")
    print(f"  Loaded: {path} ({len(content):,} chars)")
    return content


# ---------------------------------------------------------------------------
# Step 1: Analyze chainer (RLM)
# ---------------------------------------------------------------------------

class ChainerAnalysisSignature(dspy.Signature):
    """
    You are a compiler/language expert analyzing a chainer (logical reasoner)
    from its source code.

    Your goal is to produce a comprehensive, structured analysis that will
    serve as the FOUNDATION for all downstream work, e.g. a relation template
    designer or an instruction writer can rely on your analysis to produce
    correct expressions that this reasoner can evaluate.

    From the source code, reverse-engineer and document, minimally:

    1. EXPRESSION FORMAT — What do valid statements and queries look like
       when passed to the chainer's Python API (add_atom / query)?
       Show the exact syntactic template with placeholders.
       IMPORTANT: The chainer may have internal runtime commands that wrap
       these expressions — clearly distinguish between the BARE EXPRESSION
       FORMAT that external callers pass to the API vs. any internal
       runtime commands.  Downstream consumers will ONLY use the bare
       expression format, never the internal commands.
    2. BUILT-IN OPERATORS — List every built-in operator/connector the
       chainer supports.  For each, show its syntax and what it does.
    3. TRUTH VALUES — What truth value forms are supported?  How are
       uncertain numeric values represented (distributions)?
    4. RULE TEMPLATES — How are if-then rules written?  Any other rules
       or logical relations that are supported? Show their structures.
    5. QUERY PATTERNS — How are queries formed for the chainer's query()
       API?  What must be a variable vs. a constant?
    6. NAMING CONVENTIONS — What naming style, if any, does the codebase
       use for predicates, constants, variables, instances, etc.?
    7. CONSTRAINTS & PITFALLS — What syntax is NOT supported?  Common
       mistakes to avoid?
    8. QUANTIFICATION AND SCOPE — How are quantifiers (universal "all",
       existential "some", cardinal "exactly N", etc.) represented?
       Is there a dedicated quantifier operator (e.g. `ForAll`, `Exists`),
       a fuzzy/probabilistic variant, or are quantifier meanings expressed
       indirectly (e.g. via `Implication` with variables in premises, or
       via explicit witness constants)?  How is scope handled when multiple
       quantifiers or negation interact? Can they be nested? Do truth
       values play a role in quantification (e.g. quantified formulas
       carrying their own TVs)?
    9. ENTITY IDENTITY AND REPRESENTATION — Does the chainer have any
       built-in notion of entity identity, or are all symbols purely
       syntactic labels that match themselves?  How should named
       entities from natural language (e.g. "Ben", "Paris") be
       represented to avoid conflating different real-world entities
       sharing the same surface name?  Are there idioms in the source
       or tests for instance-of-class, naming relations, or similar?
       What about first/second-person pronouns ("I", "we", "you") —
       is there a convention for representing these, or should they
       be skipped when their referent is unknown?

    10. PRIMITIVES CHEAT SHEET — At the END of the analysis, add a
        consolidated quick-reference section titled "Primitives cheat
        sheet" that lists every primitive documented above.  Organize
        it by category (e.g. expression scaffolding, rule structure,
        operators/connectives, truth-value and distribution forms,
        computation/aggregation helpers, formula/reducer helpers).
        For each primitive give its canonical syntactic form (one
        line, in backticks) and a one-line semantic note.  This
        section is the authoritative cheat sheet that downstream
        pipeline stages and the deployed translator LM consult — do
        not omit it, and do not introduce primitives here that are
        not documented above.

    Be precise and derive everything from the source code.  If the source
    includes reference documentation (e.g. spec files, README), consult
    those as well.

    Output the prose analysis (with the cheat-sheet section appended) as
    clear, well-organized text in `chainer_analysis`.
    """
    chainer_source: str = dspy.InputField(
        desc="Complete source code of the chainer/reasoner"
    )
    chainer_analysis: str = dspy.OutputField(
        desc="Structured analysis of the chainer's syntax, operators, "
             "constraints, and conventions, etc."
    )


# ---------------------------------------------------------------------------
# Step 2: Enumerate linguistic phenomena
# ---------------------------------------------------------------------------

class LinguisticPhenomenaSignature(dspy.Signature):
    """
    You are a linguist enumerating the linguistic phenomena that a
    semantic parsing system for English must handle.

    Produce a comprehensive list of linguistic phenomena.  Each item
    should be a self-contained string with: the category name, a brief
    description, and 2-3 short example English sentences demonstrating it.

    Cover at minimum: entity classification, properties/attributes,
    actions and events with thematic roles (agent, patient, instrument,
    etc.), spatial relations, temporal relations, quantification
    (existential, universal, cardinality), negation, modality (ability,
    possibility, necessity, permission), causation and purpose, comparison
    (greater, lesser, equal, superlative), conjunction and disjunction,
    conditionals (if-then), propositional attitudes (believes, knows,
    wants), adverbs and manner, frequency/habitual aspect.

    This is a PURE LINGUISTICS task — do not reference any specific
    logic formalism or chainer syntax.  Focus on what English expresses,
    not how to represent it.
    """
    task: str = dspy.InputField(
        desc="Task description"
    )
    linguistic_phenomena: list[str] = dspy.OutputField(
        desc="List of linguistic phenomena, each a self-contained string "
             "with category name, description, and example sentences."
    )


# ---------------------------------------------------------------------------
# Step 3: Generate phenomenon-feature mapping
# ---------------------------------------------------------------------------

class PhenomenonFeatureMappingSignature(dspy.Signature):
    """
    You are bridging two prior bootstrap outputs into a generation
    recipe for training data.

    Given:
    - `chainer_analysis`: a structured breakdown of the chainer's
      expressive capabilities (truth-value forms, operators, rule
      templates, quantification, entity representation, etc.).
    - `linguistic_phenomena`: a list of linguistic phenomena that a
      semantic parser must handle, written in pure-linguistics terms
      with no reference to any specific logic formalism.

    Produce a phenomenon-by-phenomenon mapping that tells a downstream
    data generator (1) which chainer-specific features each phenomenon
    can or should exercise, and (2) what constraints the expected
    proofs from those examples should satisfy.

    Output format (mandatory): a numbered list, one entry per
    phenomenon, in the same order as `linguistic_phenomena`.  Each
    entry uses the following structure:

        N. <phenomenon name>
           Recap: <one-line recap of the phenomenon>
           Chainer features to exercise: <named features from
             chainer_analysis that this phenomenon naturally calls for,
             OR the literal string "(no chainer-specific features
             beyond default)" if none>
           Constraint templates: <one or more constraint templates the
             expected proofs should satisfy when generated examples
             exercise this phenomenon, OR the literal string "(none)"
             if no templates apply>

    Concentrate constraint generation on phenomena where:
    - English naturally expresses uncertainty, hedging, frequency,
      modality, or probability — the proof's truth value should
      reflect that uncertainty rather than defaulting to the chainer's
      "fully true" form (e.g. (STV 1.0 1.0)).
    - English naturally calls for universal, existential, cardinal,
      or comparative quantification — the proof should use the
      chainer's appropriate quantifier idioms.
    - English implies conditional or rule-like relationships — the
      proof should reach for the chainer's rule templates rather
      than monolithic predicates.

    Skip constraints on phenomena where adding a constraint would be
    artificial — not every phenomenon needs a chainer-feature
    constraint, and over-constraining will reduce data quality.  Mark
    those explicitly as "(no chainer-specific features beyond default)"
    and "(none)".

    Constraint templates should be:
    - Specific enough to be machine-checkable (e.g. "STV strength in
      [0.3, 0.7] for hedged frequency claims", not just "uncertain TV").
    - Reference features by NAME from chainer_analysis (e.g. "STV",
      "Implication rule", "ForAll") so the data generator and judge
      both know what to look for in the chainer's output.
    - Self-contained: one constraint per line, readable in isolation.

    Output as clear, well-structured text.  The data generator parses
    this file directly into per-phenomenon blocks as its primary
    prompt input.
    """
    chainer_analysis: str = dspy.InputField(
        desc="Structured analysis of the chainer's syntactic forms, "
             "operators, truth-value forms, etc. (output of Step 1)."
    )
    linguistic_phenomena: str = dspy.InputField(
        desc="Numbered list of linguistic phenomena (output of Step 2)."
    )
    phenomenon_feature_mapping: str = dspy.OutputField(
        desc="Per-phenomenon mapping listing the chainer features each "
             "phenomenon should exercise and constraint templates for "
             "the expected proofs."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap a chainer for semantic parsing (3 steps)"
    )
    parser.add_argument(
        "--chainer",
        default=None,
        help="Path to the chainer directory (required for Step 1, "
             "which reads the chainer source)",
    )
    parser.add_argument(
        "--from-step",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Start from this step, loading earlier outputs from disk (default: 1)",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-5.4-mini",
        help="LM identifier (default: openai/gpt-5.4-mini)",
    )
    parser.add_argument(
        "--effort",
        default="high",
        choices=["low", "medium", "high", "xhigh"],
        help="Reasoning effort (default: high)",
    )
    parser.add_argument("--output-analysis", default=str(_DEFAULTS["analysis"]))
    parser.add_argument("--output-phenomena", default=str(_DEFAULTS["phenomena"]))
    parser.add_argument("--output-mapping", default=str(_DEFAULTS["mapping"]))
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging verbosity (default: info)",
    )
    parser.add_argument(
        "--log-file",
        default="/tmp/bootstrap_chainer.log",
        help="Path to log file (default: /tmp/bootstrap_chainer.log)",
    )
    args = parser.parse_args()

    # Validate: --chainer required only when running Step 1
    if args.from_step <= 1 and args.chainer is None:
        parser.error("--chainer is required when --from-step <= 1 "
                     "(Step 1 reads the chainer source)")

    # Configure logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_file = args.log_file
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    root = logging.getLogger()
    root.setLevel(log_level)
    console_h = logging.StreamHandler()
    console_h.setLevel(log_level)
    console_h.setFormatter(fmt)
    root.addHandler(console_h)
    file_h = logging.FileHandler(log_file, mode="w")
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(fmt)
    root.addHandler(file_h)

    class _Tee:
        def __init__(self, original, log_fh):
            self._original = original
            self._log_fh = log_fh
        def write(self, data):
            self._original.write(data)
            self._log_fh.write(data)
            self._log_fh.flush()
        def flush(self):
            self._original.flush()
            self._log_fh.flush()
        def fileno(self):
            return self._original.fileno()
        def isatty(self):
            return self._original.isatty()

    _log_fobj = open(log_file, "a")
    sys.stdout = _Tee(sys.__stdout__, _log_fobj)
    sys.stderr = _Tee(sys.__stderr__, _log_fobj)

    logging.info("Log file: %s", log_file)

    if log_level > logging.DEBUG:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    _failure_logger = logging.getLogger("litellm.failures")

    def _on_failure(kwargs, response_obj, start_time, end_time):
        exc = kwargs.get("exception")
        if exc:
            chain = []
            e = exc
            while e:
                chain.append(f"  {type(e).__name__}: {e}")
                e = e.__cause__ or e.__context__
            _failure_logger.error(
                "LiteLLM call failed (model=%s):\n%s",
                kwargs.get("model", "?"),
                "\n".join(chain),
            )

    litellm.failure_callback = [_on_failure]

    lm = dspy.LM(args.model, reasoning_effort=args.effort,
                  timeout=600, num_retries=3)
    dspy.configure(lm=lm)

    paths = {
        "analysis": pathlib.Path(args.output_analysis),
        "phenomena": pathlib.Path(args.output_phenomena),
        "mapping": pathlib.Path(args.output_mapping),
    }

    from_step = args.from_step

    # Pre-declared so step 3 can load these from disk if earlier steps were skipped.
    chainer_analysis = None
    phenomena_text = None

    # =================================================================
    # Step 1: Analyze chainer (RLM)
    # =================================================================
    if from_step <= 1:
        print(f"\nReading chainer source from {args.chainer} ...")
        chainer_source = _read_chainer_source_code(pathlib.Path(args.chainer))
        print(f"  Source size: {len(chainer_source):,} characters")

        print("\n" + "=" * 60)
        print("Step 1: Analyzing chainer syntax and capabilities (RLM) ...")

        predictor = RLM(ChainerAnalysisSignature,
                        max_iterations=30, max_llm_calls=50, verbose=True)
        result = predictor(chainer_source=chainer_source)
        chainer_analysis = result.chainer_analysis
        _save_text(paths["analysis"], chainer_analysis)
        print(f"  Analysis complete ({len(chainer_analysis):,} chars)")

    # =================================================================
    # Step 2: Enumerate linguistic phenomena
    # =================================================================
    if from_step <= 2:
        print("\n" + "=" * 60)
        print("Step 2: Enumerating linguistic phenomena ...")

        phenomena_list = dspy.Predict(LinguisticPhenomenaSignature)(
            task="Enumerate all linguistic phenomena that a semantic "
                 "parsing system must handle when converting English to logic",
        ).linguistic_phenomena
        # Serialize list to numbered text for file storage and downstream string inputs
        phenomena_text = "\n\n".join(
            f"{i}. {p}" for i, p in enumerate(phenomena_list, 1)
        )
        _save_text(paths["phenomena"], phenomena_text)
        print(f"  Enumerated {len(phenomena_list)} phenomena")

    # =================================================================
    # Step 3: Generate phenomenon-feature mapping
    # =================================================================
    if from_step <= 3:
        print("\n" + "=" * 60)
        print("Step 3: Generating phenomenon-feature mapping ...")

        if chainer_analysis is None:
            chainer_analysis = _load_text(paths["analysis"], "chainer_analysis")
        if phenomena_text is None:
            phenomena_text = _load_text(paths["phenomena"], "linguistic_phenomena")

        mapping_text = dspy.Predict(PhenomenonFeatureMappingSignature)(
            chainer_analysis=chainer_analysis,
            linguistic_phenomena=phenomena_text,
        ).phenomenon_feature_mapping
        _save_text(paths["mapping"], mapping_text)
        print(f"  Mapping complete ({len(mapping_text):,} chars)")

    # =================================================================
    # Done
    # =================================================================
    print("\n" + "=" * 60)
    print("Bootstrap complete. Outputs:")
    for name, path in paths.items():
        exists = "✓" if path.exists() else " "
        print(f"  [{exists}] {path}")
    print("\nNext steps:")
    print("  - bootstrap/chainer_analysis.txt → NL2PLNModule pln_spec (via --pln-spec-file)")
    print("  - bootstrap/phenomenon_feature_mapping.txt → generate_data.py input")


if __name__ == "__main__":
    main()
