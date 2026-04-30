"""
Bootstrap a chainer for semantic parsing.

Reads the chainer's source code and generates everything needed for the
NL-to-PLN pipeline in four steps:

  1. Analyze the chainer (RLM) — reverse-engineer syntax, operators,
     constraints from the source code.  Includes a final cheat-sheet
     section listing each primitive with its canonical form.

  2. Enumerate linguistic phenomena (Predict) — comprehensive list of
     phenomena that the semantic parser must handle.

  3. Generate NL→logic conversion guidelines (RLM) — systematic rules
     for mapping linguistic phenomena to the chainer's logic format,
     plus canonical naming vocabulary for closed-class semantic relations
     (thematic roles, spatial/temporal relations, modality, attitudes,
     etc.).  RLM-based so the LM may execute code to explore the chainer
     source for verification when chainer_analysis seems insufficient.

  4. Assemble final conversion instructions (deterministic) — wraps
     conversion_guidelines into instructions.md.

Steps 1 and 3 use RLM (Recursive LM, requires Deno) to explore the
chainer source code via iterative code execution.  Step 2 uses
dspy.Predict (single LLM call).  Step 4 is deterministic (no LLM call).

Each step saves its output to a file.  Use --from-step N to skip earlier
steps and load their outputs from disk (useful for iterating on later
steps without re-running expensive earlier ones).

Usage:
    python src/bootstrap_chainer.py --chainer ../PeTTaChainer
    python src/bootstrap_chainer.py --from-step 3 --chainer ../PeTTaChainer
    python src/bootstrap_chainer.py --from-step 4
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
# Step 3: Generate conversion guidelines
# ---------------------------------------------------------------------------

class ConversionGuidelinesSignature(dspy.Signature):
    """
    You are a semantic parsing expert designing systematic English→logic
    mapping rules for a chainer/reasoner.

    You are given:
    1. A complete analysis of the chainer's syntax, operators,
       constraints, etc.
    2. A list of linguistic phenomena that a semantic parser should
       be able to handle at minimum.

    Produce CONVERSION GUIDELINES — systematic rules for mapping
    linguistic phenomena to the chainer's logic format.

    Role separation (the basis for non-redundancy):

    chainer_analysis is the authoritative reference on what the chainer
    IS — its syntactic forms, primitives, truth-value forms, constraints,
    and surface conventions.  Your guidelines are PRESCRIPTIVE — they
    document English→logic mapping disciplines that compose with whatever
    the chainer provides.  They are not a restatement of the chainer.

    Concretely:
    - Reference primitives by role/name only; never restate their
      syntactic templates, never re-enumerate which operators or forms
      exist in the chainer.
    - Use literal-syntax-bearing examples ONLY where prescriptive
      content genuinely requires them (the BAD → GOOD anti-pattern
      sections below).  Everywhere else, name what the rule produces
      structurally in prose; the consumer LLM will consult
      chainer_analysis for syntactic form.
    - Where the analysis specifies a convention or constraint, defer to
      it without restating; where the analysis is silent on something
      downstream translation needs, fill the gap (see "Translation
      conventions" below).

    Source-code fallback:

    chainer_source contains the full source of the chainer/reasoner.
    You may execute code (read specific files, search for patterns,
    try small examples) to verify a claim or fill a gap when
    chainer_analysis seems incomplete.  Use sparingly — the analysis
    is your primary reference; most guidelines are derivable from
    analysis + phenomena alone, and exhaustive re-reading wastes
    iterations.

    Mapping disciplines:

    - Predicate atomicity (the MOST CRITICAL discipline): every
      predicate name must denote a single semantic primitive — typically
      1-3 English words (a noun, verb, or short phrase).  Predicate
      names are the handles by which the reasoner composes facts and
      rules into proofs; monolithic names that bake in compound
      structures (multiple concepts, numerical constants, entity-
      specific configurations) become opaque to the reasoner's
      compositional machinery and can only participate in proofs by
      matching themselves literally.

      Anti-patterns to forbid (express each correctly using whatever
      primitives the chainer analysis documents):

        * Numbers, cardinalities, or counts embedded in the name.
          Introduce witness constants or variables and use the
          chainer's counting/comparison facilities instead.

        * Numerical thresholds or ranges embedded in the name.  Expose
          the measured value as an argument of a predicate and use the
          chainer's comparison facilities as separate premises.

        * Multi-concept concatenations where one name stands for a
          compound relationship.  Split into multiple predicates each
          denoting one concept, linked by shared variables or constants.

        * Entity-specific comparisons (pairs or triples of named
          entities fused into the name).  Express the relation between
          two first-class values using general binary or n-ary operators.

      Self-check: could this predicate appear in a position that feeds
      into further inference — as an input to another rule, as part of
      a larger pattern, as an argument to another operator?  If the
      name describes the whole answer, the reasoning has been absorbed
      into the predicate string and the reasoner has nothing further
      to do.

      Your conversion guidelines MUST dedicate a section to predicate
      atomicity with EXACTLY 4 BAD → GOOD contrastive examples — one
      per anti-pattern category listed above.  Do not produce
      additional examples beyond the 4 categories.  Synthesize the
      examples yourself by reverse-engineering plausible English
      constructions against the chainer's documented primitives; do
      not rely on canned templates, and do not reuse examples from
      this instruction.

    - Decomposition strategies: how to break complex sentences into
      multiple atomic statements, linked through whatever binding
      mechanism the chainer provides (typically shared variables or
      shared constants — defer to the analysis).  This is the
      structural complement to predicate atomicity — one applies at
      the sentence-to-atoms boundary, the other at the predicate-
      naming boundary.

    - Rule-vs-query separation: statements and queries are different
      speech acts — statements contribute to the KB, queries request
      proofs.  How the chainer surfaces this distinction (shared
      shapes, distinct constructs, or no distinction at all) is for
      chainer_analysis to document; defer to it.

      A failure mode worth guarding against (relevant where the
      chainer treats queries as targeting derivable facts rather than
      rule forms): when the English expresses universal, generic, or
      conditional content ("all X are Y", "if X then Y", "every X is
      a Y"), the LLM may mistakenly copy the rule-shaped surface form
      into the query position, asking the reasoner to prove the rule
      itself rather than to use it.  Where the chainer's analysis
      exhibits this distinction, the guidelines must state the
      separation explicitly with one contrastive BAD → GOOD example
      synthesized from the chainer's documented surface forms (add
      more only if a second anti-pattern requires a distinct positive
      form).  If the analysis indicates queries and rules share syntax
      or rule-shaped queries are first-class, follow whatever the
      analysis specifies and adapt this guidance accordingly.

    - Query granularity: whatever decomposition discipline applies to
      STATEMENTS may not apply identically to QUERIES.

      A failure mode worth guarding against: after decomposing a
      sentence into many atomic facts, the LLM emits a parallel
      battery of queries re-verifying every atomic piece.  Those facts
      are already in the KB; re-querying them dilutes the proof search
      without adding information.  Queries should target only the
      specific unknown the English question asks about.

      If the chainer's analysis indicates queries have distinct
      granularity conventions from statements, articulate that
      asymmetry with one contrastive BAD → GOOD example synthesized
      from the analysis (add more only if a second anti-pattern
      requires a distinct positive form).

    - Truth-value assignment strategies for English uncertainty: how
      to map English hedges, modals, hearsay, generics, defeasibility,
      and negation onto whatever truth-value forms the chainer
      provides.  This is the prescriptive side; the chainer's
      available truth-value forms themselves are documented in the
      analysis.

    - Phenomenon-specific mapping notes: address ONLY those phenomena
      whose mapping requires non-default treatment, has a
      counterintuitive pitfall, or needs vocabulary the disciplines
      above don't determine.  For phenomena handled correctly by the
      disciplines above (atomicity, decomposition, rule-vs-query,
      query-granularity, truth-value strategy), do NOT produce a
      per-phenomenon subsection — a single sentence saying "apply
      general rules" or omission is preferred.  The phenomena list is
      for awareness, not exhaustive enumeration.

      When a phenomenon DOES warrant its own note, write the note as
      TEXT-ONLY guidance — reference canonical predicates by name
      (drawn from the canonical relation vocabulary section below; do
      not re-list names there), and describe the structural pattern
      in prose.  Do NOT include worked literal-syntax examples in
      this section: those are exactly what an optimizer's demos will
      provide authoritatively from real successful trajectories, and
      bootstrap-imagined worked examples here may conflict with the
      names the optimizer eventually settles on.  Worked literal-
      syntax examples remain appropriate ONLY in the BAD → GOOD
      anti-pattern sections (atomicity, rule-vs-query, query-
      granularity), where the negative signal is something demos
      cannot teach.

    Translation conventions:

    These document translator-side naming choices that must be
    consistent across all translations.  Where the chainer analysis
    specifies any of them, defer to it without restating; where the
    analysis is silent, document a sensible convention here.

    - Predicate naming style for names DERIVED from English content
      (lemmatization, capitalization, compound handling).  This covers
      translator-introduced predicates, not chainer-defined primitives.
    - Variable naming conventions for translator-introduced variables.
    - Entity-id schema for resolving named entities, disambiguating
      entities sharing surface names, and handling first/second-person
      pronouns when their referent is unknown.

    Canonical relation vocabulary (closed-class):

    Document a glossary of canonical names for closed-class semantic
    relations — relations English uses routinely but the chainer
    typically does not provide as primitives (thematic roles,
    spatial/temporal relations, modality, propositional attitudes,
    evidentiality, etc.).  Phenomenon-specific mapping notes (above)
    reference these names without re-listing them.

    Format (mandatory, one block per category):

      <category name>
        `(Name1 args)` `(Name2 args)` ... [list of canonical names]
        Disambiguation notes (only where names are confusable):
        - Name1 vs Name2: [one line]

    No per-name description blocks for self-descriptive names.

    Categories to cover (typical, non-exhaustive): reference and
    identity; possession, containment, part-whole, and membership;
    thematic and event roles; spatial relations; deictic and context-
    relative reference; temporal relations; aspect, recurrence, and
    frequency; causation, purpose, reason, and means; change of state,
    creation, destruction, and existence; modality, ability,
    permission, and norms; propositional attitudes, desires, and
    plans; communication and reported content; evidentiality and
    information source; negation, absence, exclusion, and
    incompatibility; quantification, cardinality, proportions, and
    exceptions; comparison, degree, ranking, and sequence;
    collections, plurality, mass nouns, and distributivity; clause
    embedding, modification, and attachment; focus, discourse status,
    and presupposition; questions, directives, and speech acts;
    non-intersective and intensional modifiers.

    These names are NAMING SUGGESTIONS for translator consistency —
    they do NOT have chainer semantics unless the chainer analysis
    lists them as primitives.  Do not duplicate any name the analysis
    lists as a primitive; defer to the analysis for those.

    These guidelines act as a "style guide" applied alongside
    chainer_analysis to ensure consistency across all semantically
    parsed sentences.

    All literal-syntax content must use the exact forms documented in
    the chainer analysis.  Do not invent forms not documented there,
    otherwise they won't work.

    Output as clear, well-organized text.
    """
    chainer_analysis: str = dspy.InputField(
        desc="Complete analysis of the chainer's syntax and capabilities"
    )
    linguistic_phenomena: str = dspy.InputField(
        desc="Comprehensive list of linguistic phenomena to handle"
    )
    chainer_source: str = dspy.InputField(
        desc="Complete source code of the chainer/reasoner.  Available "
             "for optional verification or filling gaps in chainer_analysis; "
             "the analysis is your primary reference, not this."
    )
    conversion_guidelines: str = dspy.OutputField(
        desc="Systematic rules for mapping linguistic phenomena to the "
             "chainer's logic format."
    )


# ---------------------------------------------------------------------------
# Step 4: Assemble final instructions (deterministic, no LLM)
# ---------------------------------------------------------------------------

def _assemble_instructions(conversion_guidelines: str) -> str:
    """
    Wrap conversion_guidelines into instructions.md.

    chainer_analysis is intentionally NOT included here — it is the
    chainer-agnostic counterpart of PeTTaChainer's LLM_RULE_SPEC.md /
    LANGUAGE_SPEC.md and is meant to flow into NL2PLNModule's `pln_spec`
    input field at deployment time, occupying a distinct semantic slot
    from the rules in this file (and avoiding token duplication when
    both flow into the same LM call).  The chainer_analysis already
    includes a primitives cheat sheet section; downstream LMs reach
    for primitive names there, not here.
    """
    return (
        "# NL to logic conversion instructions\n\n"
        "You are converting English natural language (sentences or questions) "
        "into logical expressions (statements or queries) for a chainer/reasoner.  "
        "The chainer's syntax and semantics reference is provided separately as "
        "the `pln_spec` input (see `chainer_analysis.txt`).\n\n"
        + conversion_guidelines
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap a chainer for semantic parsing (4 steps)"
    )
    parser.add_argument(
        "--chainer",
        default=None,
        help="Path to the chainer directory (required for Steps 1 and 3, "
             "which read the chainer source)",
    )
    parser.add_argument(
        "--from-step",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
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

    # Validate: --chainer required for Steps 1 and 3 (both read source)
    if args.from_step <= 3 and args.chainer is None:
        parser.error("--chainer is required when --from-step <= 3 "
                     "(Steps 1 and 3 read the chainer source)")

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

    # Load chainer source once if any step that reads it (1, 3) will run.
    chainer_source = None
    if from_step <= 3:
        print(f"\nReading chainer source from {args.chainer} ...")
        chainer_source = _read_chainer_source_code(pathlib.Path(args.chainer))
        print(f"  Source size: {len(chainer_source):,} characters")

    # =================================================================
    # Step 1: Analyze chainer (RLM)
    # =================================================================
    if from_step <= 1:
        print("\n" + "=" * 60)
        print("Step 1: Analyzing chainer syntax and capabilities (RLM) ...")

        predictor = RLM(ChainerAnalysisSignature,
                        max_iterations=30, max_llm_calls=50, verbose=True)
        result = predictor(chainer_source=chainer_source)
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
    # Step 3: Generate conversion guidelines (RLM)
    # =================================================================
    if from_step <= 3:
        print("\n" + "=" * 60)
        print("Step 3: Generating conversion guidelines (RLM) ...")

        predictor = RLM(ConversionGuidelinesSignature,
                        max_iterations=30, max_llm_calls=50, verbose=True)
        result = predictor(
            chainer_analysis=ctx["chainer_analysis"],
            linguistic_phenomena=ctx["linguistic_phenomena"],
            chainer_source=chainer_source,
        )
        ctx["conversion_guidelines"] = result.conversion_guidelines
        _save_text(paths["guidelines"], ctx["conversion_guidelines"])

    # =================================================================
    # Step 4: Assemble final instructions (deterministic, no LLM)
    # =================================================================
    if from_step <= 4:
        print("\n" + "=" * 60)
        print("Step 4: Assembling conversion instructions ...")

        instructions = _assemble_instructions(
            conversion_guidelines=ctx["conversion_guidelines"],
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
    print("\nFor deployment, pass instructions.md as the signature instruction")
    print("AND chainer_analysis.txt separately as the NL2PLNModule pln_spec")
    print("input (e.g. via --pln-spec-file).  The chainer_analysis includes the")
    print("primitives cheat sheet; instructions.md carries the prescriptive")
    print("guidelines (with canonical relation vocabulary).")
    print("Review the artifacts, then run the optimizer.")


if __name__ == "__main__":
    main()
