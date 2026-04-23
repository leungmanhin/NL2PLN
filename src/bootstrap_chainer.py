"""
Bootstrap a chainer for semantic parsing.

Reads the chainer's source code and generates everything needed for the
NL-to-PLN pipeline in five steps:

  1. Analyze the chainer (RLM) — reverse-engineer syntax, operators,
     constraints from the source code.

  2. Enumerate linguistic phenomena (Predict) — comprehensive list of
     English constructs that the semantic parser must handle.

  3. Generate NL→logic conversion guidelines (Predict) — systematic
     rules for mapping each phenomenon to the chainer's logic format.

  4. Generate chainer primitives and suggested vocabulary (Predict) —
     split into two outputs: the real chainer primitives (with semantic
     backing in the chainer's evaluation engine) vs. open-class/closed-
     class vocabulary (naming suggestions for the LLM, not primitives).

  5. Assemble final conversion instructions (deterministic) — concatenates
     the chainer analysis, primitives, guidelines, and vocabulary into
     instructions.md.

Step 1 uses RLM (Recursive LM, requires Deno) to explore the chainer
source code via iterative code execution.  Steps 2-4 use dspy.Predict
(single LLM call each).  Step 5 is deterministic (no LLM call).

Each step saves its output to a file.  Use --from-step N to skip earlier
steps and load their outputs from disk (useful for iterating on later
steps without re-running expensive earlier ones).

Usage:
    python src/bootstrap_chainer.py --chainer ../PeTTaChainer
    python src/bootstrap_chainer.py --from-step 4 --chainer ../PeTTaChainer
    python src/bootstrap_chainer.py --from-step 5
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
    "analysis": _PROJECT_ROOT / "chainer_analysis.txt",
    "phenomena": _PROJECT_ROOT / "linguistic_phenomena.txt",
    "guidelines": _PROJECT_ROOT / "conversion_guidelines.txt",
    "primitives": _PROJECT_ROOT / "chainer_primitives.txt",
    "vocabulary": _PROJECT_ROOT / "suggested_vocabulary.txt",
    "instructions": _PROJECT_ROOT / "instructions.md",
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
    serve as the FOUNDATION for all downstream work — a relation template
    designer and an instruction writer will rely on your analysis to produce
    correct logical expressions.

    From the source code, reverse-engineer and document, minimally:

    1. EXPRESSION FORMAT — What do valid statements and queries look like
       when passed to the chainer's Python API (add_atom / query)?
       Show the exact syntactic template with placeholders.
       IMPORTANT: The chainer may have internal runtime commands that wrap
       these expressions — clearly distinguish between the BARE EXPRESSION
       FORMAT that external callers pass to the API vs. any internal
       runtime commands.  Downstream consumers will ONLY use the bare
       expression format, never the internal commands.
    2. BUILT-IN OPERATORS — List every built-in operator/combinator the
       chainer supports.  For each, show its syntax and what it does.
    3. TRUTH VALUES — What truth value forms are supported?  How are
       uncertain numeric values represented (distributions)?
    4. RULE TEMPLATES — How are if-then rules written?  Show the exact
       structure.
    5. QUERY PATTERNS — How are queries formed for the chainer's query()
       API?  What must be a variable vs. a constant?
    6. NAMING CONVENTIONS — What naming style does the codebase use for
       predicates, constants, variables?
    7. CONSTRAINTS & PITFALLS — What syntax is NOT supported?  Common
       mistakes to avoid?  Explicitly note: expressions must be in the
       bare format accepted by the API, NOT wrapped in internal runtime
       commands.
    8. QUANTIFICATION AND SCOPE — How are quantifiers (universal "all",
       existential "some", cardinal "exactly N", etc.) represented?
       Is there a dedicated quantifier operator (e.g. `ForAll`,
       `Exists`), or are quantifier meanings expressed indirectly (e.g.
       via `Implication` with variables in premises, or via explicit
       witness constants)?  How is scope handled when multiple
       quantifiers or negation interact?
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

    Be precise and derive everything from the source code.  If the source
    includes reference documentation (e.g. spec files, README), use those
    as references too.

    Output the prose analysis as clear, well-organized text in
    `chainer_analysis`.
    """
    chainer_source: str = dspy.InputField(
        desc="Complete source code of the chainer/reasoner"
    )
    chainer_analysis: str = dspy.OutputField(
        desc="Structured analysis of the chainer's syntax, operators, "
             "constraints, and conventions."
    )


# ---------------------------------------------------------------------------
# Step 2: Enumerate linguistic phenomena
# ---------------------------------------------------------------------------

class LinguisticPhenomenaSignature(dspy.Signature):
    """
    You are a linguist enumerating the kinds of English constructs that
    a semantic parsing system must handle.

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
# Step 3: Generate conversion guidelines
# ---------------------------------------------------------------------------

class ConversionGuidelinesSignature(dspy.Signature):
    """
    You are a semantic parsing expert designing systematic rules for
    converting English sentences into logical expressions for a specific
    chainer/reasoner.

    You are given:
    1. A complete analysis of the chainer's syntax, operators, and
       constraints
    2. A comprehensive list of linguistic phenomena to handle

    Produce CONVERSION GUIDELINES — systematic rules for mapping each
    linguistic phenomenon to the chainer's logic format.
    Cover at minimum:

    - Naming conventions: how to derive predicate names from English
      words (lemmatization, capitalization style, etc.)
    - Entity representation and identity: how to map named entities
      from English to logic symbols (e.g. instance-of-class with a
      separate naming relation vs. using the name directly as a
      symbol); how to disambiguate different real-world entities that
      share the same surface name; how to handle first/second-person
      pronouns ("I", "we", "you") when their referent is unknown.
      Follow whatever the chainer analysis says about identity
      semantics.
    - Quantification and scope: how to express universal ("all",
      "every"), existential ("some", "there is"), and cardinal
      ("exactly N") quantifiers using the chainer's available
      machinery (dedicated operator vs. rule-with-variables vs.
      witness constants).  How to handle scope interactions with
      negation and other quantifiers.
    - Predicate atomicity (the MOST CRITICAL discipline): every
      predicate name must denote a single semantic primitive — typically
      1-3 English words (a noun, verb, or short phrase). Predicate
      names are the handles by which the reasoner composes facts and
      rules into proofs; monolithic names that bake in compound
      structures (multiple concepts, numerical constants, entity-
      specific configurations) become opaque to the reasoner's
      compositional machinery and can only participate in proofs by
      matching themselves literally. Consult the chainer analysis for
      exactly how predicate names are used during proof construction.

      Anti-patterns to forbid (consult the chainer analysis for which
      built-in operators and truth-value constructors are available to
      express each correctly):

        * Numbers, cardinalities, or counts embedded in the name.
          Introduce witness constants or variables and use the chainer's
          comparison or counting primitives instead.

        * Numerical thresholds or ranges embedded in the name. Expose
          the measured value as an argument of a predicate and use the
          chainer's comparison operators as separate premises.

        * Multi-concept concatenations where one name stands for a
          compound relationship. Split into multiple predicates each
          denoting one concept, linked by shared variables or constants.

        * Entity-specific comparisons (pairs or triples of named
          entities fused into the name). Express the relation between
          two first-class values using general binary or n-ary operators.

      Self-check: could this predicate appear in a position that feeds
      into further inference — as an input to another rule, as part of
      a larger pattern, as an argument to another operator? If the name
      describes the whole answer, the reasoning has been absorbed into
      the predicate string and the reasoner has nothing further to do.

      Your conversion guidelines MUST dedicate a section to predicate
      atomicity with concrete BAD → GOOD contrastive examples.
      Synthesize the examples yourself by reverse-engineering plausible
      English constructions against the chainer's documented primitives;
      do not rely on canned templates, and do not reuse examples from
      this instruction.

    - Decomposition strategies: how to break complex sentences into
      multiple atomic statements, linked through whatever binding
      mechanism the chainer analysis documents (typically shared
      variables or shared constants). This is the structural
      complement to predicate atomicity — one applies at the
      sentence-to-atoms boundary, the other at the predicate-naming
      boundary.

    - Query semantics: statements and queries are different speech
      acts — statements contribute to the KB, queries request proofs.
      Consult the chainer analysis for how the chainer distinguishes
      these roles: whether they share surface shapes or use distinct
      constructs, what a query is permitted to target, and how the
      reasoner treats each.

      A failure mode worth guarding against (relevant wherever the
      chainer analysis indicates queries target derivable facts rather
      than rule forms): when the English expresses universal, generic,
      or conditional content ("all X are Y", "if X then Y", "every X
      is a Y"), the LLM may mistakenly copy the rule-shaped surface
      form into the query position, asking the reasoner to prove the
      rule itself rather than to use it. Wherever the chainer analysis
      documents this distinction, the guidelines must state the
      rule-vs-query separation explicitly, with contrastive BAD → GOOD
      examples synthesized from the chainer's documented surface
      forms. (If the chainer permits querying rule forms directly or
      uses identical syntax for rules and queries, follow whatever the
      chainer analysis specifies.)

    - Query granularity: whatever decomposition discipline applies to
      STATEMENTS may not apply identically to QUERIES. Consult the
      chainer analysis for how queries are consumed (single-shot
      unification, multi-step search, interactive refinement, etc.)
      and derive appropriate granularity guidance from that operational
      model.

      A failure mode worth guarding against: after decomposing a
      sentence into many atomic facts, the LLM emits a parallel
      battery of queries re-verifying every atomic piece. Those facts
      are already in the KB; re-querying them dilutes the proof search
      without adding information. Queries should target only the
      specific unknown the English question asks about.

      If the chainer analysis indicates queries have distinct
      granularity conventions from statements, articulate that
      asymmetry in the guidelines with contrastive BAD → GOOD examples
      synthesized from the chainer analysis.

    - When to use built-in operators vs. custom predicates
    - Variable naming conventions
    - Truth value assignment strategies
    - How to handle each linguistic phenomenon category from the
      phenomena list

    These guidelines will be used as a "style guide" to ensure
    consistency across all semantically parsed sentences.

    All guidelines must use the exact syntax documented in the chainer
    analysis.  Do not invent syntax not documented there otherwise it
    won't work.

    Output as clear, well-organized text.
    """
    chainer_analysis: str = dspy.InputField(
        desc="Complete analysis of the chainer's syntax and capabilities"
    )
    linguistic_phenomena: str = dspy.InputField(
        desc="Comprehensive list of linguistic phenomena to handle"
    )
    conversion_guidelines: str = dspy.OutputField(
        desc="Systematic rules for mapping English constructs to the "
             "chainer's logic format.  Plain text."
    )


# ---------------------------------------------------------------------------
# Step 4: Generate chainer primitives + suggested vocabulary
# ---------------------------------------------------------------------------

class RelationTemplatesSignature(dspy.Signature):
    """
    You are producing TWO separate bodies of content for a semantic parsing
    pipeline that converts natural language to logical expressions against
    a specific chainer/reasoner.

    The distinction between the two outputs is CRITICAL and must be
    preserved — conflating them causes downstream LLMs to treat
    vocabulary-style suggestions as if they were chainer primitives, which
    produces KB facts the reasoner cannot unify or chain through.

    1. chainer_primitives:
       The REAL primitives documented in the chainer analysis — query
       scaffolding, built-in operators, truth-value constructors, and any
       other constructs that have SEMANTIC BACKING in the chainer's
       evaluation engine. Names listed here will be used VERBATIM by
       downstream LLMs, and the reasoner will apply meaning to them during
       proof search. Include only what the chainer analysis documents as a
       built-in or operational construct.

    2. suggested_vocabulary:
       Open-class predicate template families AND closed-class canonical
       relation names for domain concepts (e.g., canonical names for
       thematic roles, spatial relations, temporal relations, modality,
       propositional attitudes, etc.). These are NAMING SUGGESTIONS for
       the LLM to maintain consistency across translations. They do NOT
       have chainer semantics unless the chainer analysis explicitly
       documents them as primitives. Mark this clearly with a preamble
       in the output.

    Organize each output with clear section headers. Use the exact syntax
    from the chainer analysis. Be general-purpose (not domain-specific)
    so the templates can combine to cover the full phenomena list. Both
    outputs must be plain text.
    """
    chainer_analysis: str = dspy.InputField(
        desc="Complete analysis of the chainer's syntax and capabilities"
    )
    linguistic_phenomena: str = dspy.InputField(
        desc="Comprehensive list of linguistic phenomena to cover"
    )
    conversion_guidelines: str = dspy.InputField(
        desc="Systematic conversion rules (naming, decomposition, etc.)"
    )
    chainer_primitives: str = dspy.OutputField(
        desc="The real chainer primitives — query scaffolding, built-in "
             "operators, truth-value constructors — that have semantic "
             "backing in the chainer.  Use these verbatim.  Plain text, "
             "organized by category."
    )
    suggested_vocabulary: str = dspy.OutputField(
        desc="Open-class predicate template families and closed-class "
             "canonical relation names for domain concepts.  NAMING "
             "SUGGESTIONS only, NOT chainer primitives.  Include a clear "
             "preamble stating this.  Plain text, organized by category."
    )


# ---------------------------------------------------------------------------
# Step 5: Assemble final instructions (deterministic, no LLM)
# ---------------------------------------------------------------------------

def _assemble_instructions(chainer_analysis: str, conversion_guidelines: str,
                           chainer_primitives: str,
                           suggested_vocabulary: str) -> str:
    """Concatenate all four artifacts into a single instruction document."""
    return (
        "# NL to logic conversion instructions\n\n"
        "You are converting English natural language (sentences or questions) "
        "into logical expressions (statements or queries) for a chainer/reasoner.\n\n"
        "## Chainer syntax reference\n\n"
        + chainer_analysis
        + "\n\n---\n\n"
        "## Chainer primitives\n\n"
        + chainer_primitives
        + "\n\n---\n\n"
        "## Conversion guidelines\n\n"
        + conversion_guidelines
        + "\n\n---\n\n"
        "## Suggested vocabulary\n\n"
        + suggested_vocabulary
    )


# ---------------------------------------------------------------------------
# Step 4 orchestrator
# ---------------------------------------------------------------------------

def _run_step4(chainer_analysis: str, linguistic_phenomena: str,
               conversion_guidelines: str) -> tuple[str, str]:
    """
    Generate the two template outputs (chainer primitives + suggested
    vocabulary).  Returns them as a (primitives, vocabulary) pair.
    """
    print("  Generating chainer primitives + suggested vocabulary ...")
    result = dspy.Predict(RelationTemplatesSignature)(
        chainer_analysis=chainer_analysis,
        linguistic_phenomena=linguistic_phenomena,
        conversion_guidelines=conversion_guidelines,
    )
    primitives = result.chainer_primitives
    vocabulary = result.suggested_vocabulary
    print(f"  Generated primitives ({len(primitives):,} chars), "
          f"vocabulary ({len(vocabulary):,} chars)")
    return primitives, vocabulary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap a chainer for semantic parsing (5 steps)"
    )
    parser.add_argument(
        "--chainer",
        default=None,
        help="Path to the chainer directory (required for Step 1)",
    )
    parser.add_argument(
        "--from-step",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
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
    parser.add_argument("--output-guidelines", default=str(_DEFAULTS["guidelines"]))
    parser.add_argument("--output-primitives", default=str(_DEFAULTS["primitives"]))
    parser.add_argument("--output-vocabulary", default=str(_DEFAULTS["vocabulary"]))
    parser.add_argument("--output-instructions", default=str(_DEFAULTS["instructions"]))
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

    # Validate: --chainer required for Step 1
    if args.from_step <= 1 and args.chainer is None:
        parser.error("--chainer is required when running Step 1")

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
        "guidelines": pathlib.Path(args.output_guidelines),
        "primitives": pathlib.Path(args.output_primitives),
        "vocabulary": pathlib.Path(args.output_vocabulary),
        "instructions": pathlib.Path(args.output_instructions),
    }

    from_step = args.from_step
    ctx = {}  # accumulated context: name -> content string

    # Load prerequisites for --from-step
    if from_step > 1:
        print(f"\nStarting from Step {from_step}, loading earlier outputs ...")
    if from_step >= 2:
        ctx["chainer_analysis"] = _load_text(paths["analysis"], "chainer_analysis")
    if from_step >= 3:
        ctx["linguistic_phenomena"] = _load_text(paths["phenomena"], "linguistic_phenomena")
    if from_step >= 4:
        ctx["conversion_guidelines"] = _load_text(paths["guidelines"], "conversion_guidelines")
    if from_step >= 5:
        ctx["chainer_primitives"] = _load_text(paths["primitives"], "chainer_primitives")
        ctx["suggested_vocabulary"] = _load_text(paths["vocabulary"], "suggested_vocabulary")

    # =================================================================
    # Step 1: Analyze chainer (RLM)
    # =================================================================
    if from_step <= 1:
        print("\n" + "=" * 60)
        print("Step 1: Analyzing chainer syntax and capabilities (RLM) ...")
        source = _read_chainer_source_code(pathlib.Path(args.chainer))
        print(f"  Source size: {len(source):,} characters")

        predictor = RLM(ChainerAnalysisSignature,
                        max_iterations=60, max_llm_calls=90, verbose=True)
        result = predictor(chainer_source=source)
        ctx["chainer_analysis"] = result.chainer_analysis
        _save_text(paths["analysis"], ctx["chainer_analysis"])
        print(f"  Analysis complete ({len(ctx['chainer_analysis']):,} chars)")

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
        ctx["linguistic_phenomena"] = "\n\n".join(
            f"{i}. {p}" for i, p in enumerate(phenomena_list, 1)
        )
        _save_text(paths["phenomena"], ctx["linguistic_phenomena"])
        print(f"  Enumerated {len(phenomena_list)} phenomena")

    # =================================================================
    # Step 3: Generate conversion guidelines
    # =================================================================
    if from_step <= 3:
        print("\n" + "=" * 60)
        print("Step 3: Generating conversion guidelines ...")

        ctx["conversion_guidelines"] = dspy.Predict(ConversionGuidelinesSignature)(
            chainer_analysis=ctx["chainer_analysis"],
            linguistic_phenomena=ctx["linguistic_phenomena"],
        ).conversion_guidelines
        _save_text(paths["guidelines"], ctx["conversion_guidelines"])

    # =================================================================
    # Step 4: Generate chainer primitives + suggested vocabulary
    # =================================================================
    if from_step <= 4:
        print("\n" + "=" * 60)
        print("Step 4: Generating chainer primitives + suggested vocabulary ...")

        primitives, vocabulary = _run_step4(
            chainer_analysis=ctx["chainer_analysis"],
            linguistic_phenomena=ctx["linguistic_phenomena"],
            conversion_guidelines=ctx["conversion_guidelines"],
        )
        ctx["chainer_primitives"] = primitives
        ctx["suggested_vocabulary"] = vocabulary
        _save_text(paths["primitives"], ctx["chainer_primitives"])
        _save_text(paths["vocabulary"], ctx["suggested_vocabulary"])

    # =================================================================
    # Step 5: Assemble final instructions (deterministic, no LLM)
    # =================================================================
    if from_step <= 5:
        print("\n" + "=" * 60)
        print("Step 5: Assembling conversion instructions ...")

        instructions = _assemble_instructions(
            chainer_analysis=ctx["chainer_analysis"],
            conversion_guidelines=ctx["conversion_guidelines"],
            chainer_primitives=ctx["chainer_primitives"],
            suggested_vocabulary=ctx["suggested_vocabulary"],
        )
        _save_text(paths["instructions"], instructions)

    # =================================================================
    # Done
    # =================================================================
    print("\n" + "=" * 60)
    print("Bootstrap complete. Outputs:")
    for name, path in paths.items():
        exists = "✓" if path.exists() else " "
        print(f"  [{exists}] {path}")
    print("\nThe primitives and suggested vocabulary are baked into the instructions file.")
    print("Review it, then run the optimizer.")


if __name__ == "__main__":
    main()
