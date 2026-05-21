"""
Colab demo: generate a contributor-oriented software spec for any repo
using RLM with a host-side section accumulator.

Adapted from Step 1 of src/bootstrap_chainer.py.  Same RLM harness
(iterative, tool-using LM agent exploring source code), but we work
around the SUBMIT/FinalOutput payload size limit by having the LM
deliver the spec SECTION-BY-SECTION through a custom host-side tool
(`append_section`) rather than through the final SUBMIT.

  - Exploration is unconstrained: the LM can use as many iterations
    as it wants up to MAX_ITERATIONS.
  - Each `append_section` call is a small JSON-RPC round-trip, well
    inside the Deno/Pyodide line-buffer limit.
  - The final SUBMIT only carries a tiny "done" marker.
  - The full spec is assembled host-side from the accumulated
    sections, in the order the LM first wrote them (dict insertion
    order).  Calling `append_section` again with the same name
    overwrites, so the LM can revise earlier sections as it learns
    more.

Other design choices kept from the earlier iteration:
  - The spec prompt lives in its own cell (SPEC_INSTRUCTIONS) so you
    can iterate on it without touching the signature shell.
  - No hardcoded file-extension whitelist — every UTF-8-decodable
    file is a candidate, and a lightweight LLM classifier picks
    which ones belong in the context.

Each `# %%` block maps to one Colab cell.  Run them top to bottom.

Requirements:
  - A repo to analyze (public URL or already cloned on the runtime)
  - OPENAI_API_KEY (or any other LiteLLM-compatible provider) in
    Colab's Secrets panel (the key icon on the left sidebar)
  - Deno — RLM spawns a sandboxed JS/TS runtime for its tool loop
"""

# %% [cell 1 — install dependencies + Deno]
# fmt: off
# !pip install -q "dspy>=3.1.3" litellm
# !curl -fsSL https://deno.land/install.sh | sh -s -- -y
import os
os.environ["PATH"] = f"/root/.deno/bin:{os.environ.get('PATH', '')}"
# Verify Deno is on PATH (should print a version line):
# !deno --version
# fmt: on


# %% [cell 2 — API key from Colab secrets]
import os
from google.colab import userdata
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
# If using OpenRouter / Cerebras / etc., set the corresponding env var here.


# %% [cell 3 — clone the target repo]
# Replace with whatever repo you want to document.
REPO_URL = "https://github.com/trueagi-io/PeTTaChainer.git"
REPO_DIR = "/content/target_repo"
# !rm -rf {REPO_DIR} && git clone --depth 1 {REPO_URL} {REPO_DIR}


# %% [cell 4 — config]
MODEL    = "openai/gpt-5.4-mini"   # any LiteLLM id; stronger models give richer specs
EFFORT   = "high"                  # reasoning_effort: low/medium/high/xhigh
OUT_PATH = "/content/contributor_spec.md"

MAX_ITERATIONS = 60   # RLM agent turns — let the LM explore freely
MAX_LLM_CALLS  = 90   # hard cap on LLM invocations across the loop


# %% [cell 5 — collect all text files (no extension filter)]
import pathlib

# Dirs whose contents are never source: VCS metadata, caches, build
# outputs, vendored deps.  Still filtered by name because these trees
# can be enormous and would dominate the classifier's input otherwise.
_SKIP_DIRS = {".git", "__pycache__", ".mypy_cache", ".pytest_cache",
              "node_modules", ".tox", ".eggs", "venv", ".venv",
              "dist", "build", "target", ".next", ".cache"}


def collect_text_files(repo_dir: str) -> dict[str, str]:
    """Return {relative_path: content} for every file that decodes as UTF-8.

    No extension whitelist — the LLM classifier in a later cell decides
    which of these actually belong in the spec.  The 4 KB probe is a
    cheap binary filter; text files read fine after.
    """
    root = pathlib.Path(repo_dir)
    out: dict[str, str] = {}
    for f in sorted(root.rglob("*")):
        if any(p in _SKIP_DIRS or p.endswith(".egg-info") for p in f.parts):
            continue
        if not f.is_file():
            continue
        try:
            with open(f, "rb") as fh:
                probe = fh.read(4096)
            probe.decode("utf-8")
            content = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        out[str(f.relative_to(root))] = content
    if not out:
        raise FileNotFoundError(f"No text files found under {root}")
    return out


all_files = collect_text_files(REPO_DIR)
print(f"Collected {len(all_files)} text files "
      f"({sum(len(c) for c in all_files.values()):,} chars total)")


# %% [cell 6 — the spec prompt (edit this freely to iterate)]
SPEC_INSTRUCTIONS = """
You are a senior software engineer writing a SPECIFICATION DOCUMENT
for a repository, targeted at a new external contributor who has
just discovered the project on GitHub and wants to make their first
meaningful contribution.

Reverse-engineer the spec from the source code provided.  Be
concrete and derived from the code — cite files, classes, and
functions by name; quote short relevant snippets where they
clarify a point; and explain the REASON behind design choices
wherever the code reveals it (comments, docstrings, module
structure, or dependency patterns).

Your output must cover, at minimum, these ten sections:

1. PURPOSE & SCOPE
   What this project does, who it's for, the problem it solves.
   If the project belongs to a larger ecosystem (upstream deps,
   sister repos, a research tradition), briefly situate it.

2. HIGH-LEVEL ARCHITECTURE
   The major components (modules, packages, subsystems) and how
   they interact.  An ASCII diagram or a clearly labeled component
   list is fine.  Name concrete entry points (CLI commands, main
   functions, public API surfaces).

3. MODULE / DIRECTORY LAYOUT
   A walk through the top-level directories explaining what each
   one owns and where to look for a given kind of change.

4. KEY ABSTRACTIONS & DATA FLOW
   The most important classes, interfaces, or data types — what
   they represent and how values flow between them during a
   typical operation.  If there is a "golden path" (a request
   enters here, gets transformed there, exits there), trace it.

5. BUILD / RUN / TEST
   Exact commands to install dependencies, build, run, and test,
   derived from config files (pyproject.toml, package.json,
   Makefile, Cargo.toml, requirements.txt, etc.) and READMEs.
   Flag any non-standard requirements (system deps, specific
   toolchain versions, native libraries).

6. CONVENTIONS
   Coding style, naming conventions, patterns the codebase
   consistently uses (dependency injection? event bus? a
   particular error-handling idiom? a specific testing style?).
   Whatever a new PR will be held to.

7. HOW TO CONTRIBUTE
   A contributor-oriented walkthrough grounded in the ACTUAL
   extension points of this codebase: how to add a feature in the
   spirit of the existing code, how to fix a bug, how to extend
   the most commonly extended surfaces (e.g., "to add a new X,
   write a class in Y/ that implements Z, register it in W").
   Derive this from patterns you observe — do not invent generic
   advice.

8. TESTING
   How tests are organized, what kinds exist (unit, integration,
   property, golden), how to add one, and where coverage looks
   thin.

9. KNOWN GAPS & GOOD FIRST ISSUES
   Look for TODO/FIXME/XXX comments, stubbed-out branches, or
   thinly covered modules.  Call out 3-5 concrete places a new
   contributor could help.

10. PITFALLS & NON-OBVIOUS CONSTRAINTS
    Surprising invariants, footguns, constraints the code assumes
    but doesn't always enforce, past-decision residue.  The kind
    of thing a new contributor would step on without being told.

The final document is Markdown.  Use headings, fenced code blocks
for examples, and short, specific sentences.  Every claim must be
anchored in the provided source — no generic software-engineering
platitudes.
""".strip()


# %% [cell 6b — section-by-section delivery protocol]
# The protocol uses a HOST-SIDE tool (registered in cell 9 and passed
# to RLM) so that each section's content rides a small JSON-RPC call
# rather than the size-limited SUBMIT payload.  The LM can take as
# many iterations as it wants; only the final SUBMIT marker goes
# through the FinalOutput channel.
SECTION_PROTOCOL = """
DELIVERY PROTOCOL — read carefully, this is how you ship the spec:

You do NOT pass the spec content via SUBMIT.  The SUBMIT channel
has a strict payload size limit that a full spec would break.

Instead, you have a dedicated tool registered on the host:

    append_section(name: str, content: str) -> str

It saves ONE named section of the spec on the host and returns a
short status string (number of sections saved so far, total chars).

Use it like this:

    status = append_section(
        name="PURPOSE & SCOPE",
        content="This project is ... (full Markdown body here)",
    )
    print(status)

Rules:

  - Call `append_section` once per section of the spec.  There are
    ten sections listed in the outline above; emit all ten.
  - Use short, canonical section names (e.g. "PURPOSE & SCOPE",
    "HIGH-LEVEL ARCHITECTURE").  The name you pass becomes the
    Markdown heading in the final document.
  - The `content` is the section BODY — Markdown prose, bullets,
    fenced code blocks — WITHOUT a repeated heading at the top.
  - Calling `append_section` again with the same `name` OVERWRITES
    the previous content.  Use this to revise a section after
    further investigation.
  - Sections are rendered in the order they were FIRST added.  Plan
    your first-write order accordingly, or use later revises to
    shape the final flow.
  - Write each section as soon as you have enough to write it.  Do
    not accumulate ten sections' worth of content in your head
    before committing — incremental writes are resilient to
    iteration-budget exhaustion, and the tool's status response
    confirms each save.

Exploration budget:

  - You have MAX_ITERATIONS RLM turns and MAX_LLM_CALLS sub-LLM
    calls total.  Use them as you see fit.  There is no
    perfectionism penalty — but note that every unused iteration
    was an opportunity to write or revise a section.

When to SUBMIT:

  - After all ten sections have been written via append_section,
    call `SUBMIT("done")`.  The host will read the accumulated
    sections from its own state and assemble the final Markdown
    document.
  - If you run out of iterations before all sections are written,
    whatever was saved via append_section is still available — the
    host will render the partial spec.  So err on the side of
    writing something for every section early, then going back to
    revise.
""".strip()


# %% [cell 7 — signatures]
import dspy


class FileClassifierSignature(dspy.Signature):
    """
    You are filtering a repository's file listing down to the subset
    a new contributor should read to understand the project and make
    meaningful PRs.

    INCLUDE: source code in any language, tests, documentation
    (READMEs, docs/, design notes), configuration that affects build
    or runtime (pyproject.toml, package.json, Makefile, Dockerfile,
    CI workflow YAML, etc.), examples, small data files used in
    tests.

    EXCLUDE: lockfiles (package-lock.json, yarn.lock, uv.lock,
    poetry.lock, Pipfile.lock, Cargo.lock), minified or generated
    assets, large data dumps, vendored third-party code, pre-built
    artifacts that happen to decode as text, auto-generated
    boilerplate whose content carries no architectural signal.

    When in doubt, include — an over-inclusive list is cheaper than
    a missing file.

    Return exact paths from the input list, no size hints, no
    commentary.
    """
    files: list[str] = dspy.InputField(
        desc="Candidate file paths relative to the repo root, each "
             "annotated with size."
    )
    selected: list[str] = dspy.OutputField(
        desc="Subset of the input paths that should be fed to the "
             "repo-analysis LLM.  Plain paths, matching the input."
    )


class SpecBuilderSignature(dspy.Signature):
    # Intentionally no docstring — bound dynamically to
    # SPEC_INSTRUCTIONS + SECTION_PROTOCOL in cell 9.
    #
    # The `status` output field is only a small acknowledgement
    # marker.  The actual spec content is delivered via
    # `append_section()` (a host-side tool) and assembled after the
    # run from the host-side dict.
    repo_source: str = dspy.InputField(
        desc="Concatenated source of the files the classifier selected."
    )
    status: str = dspy.OutputField(
        desc="Short acknowledgement (e.g. 'done') once every section "
             "has been written via append_section()."
    )


# %% [cell 8 — classify files + build the source blob]
dspy.configure(lm=dspy.LM(MODEL, reasoning_effort=EFFORT,
                          timeout=600, num_retries=3))

# One line per file with a size hint so the classifier can weigh cost
# when deciding borderline cases.
listing = [f"{p}  ({len(c):,} chars)" for p, c in all_files.items()]
print(f"Classifying {len(listing)} candidate files ...")
classified = dspy.Predict(FileClassifierSignature)(files=listing)


def _strip_size_hint(s: str) -> str:
    s = s.strip()
    if s.endswith(")") and "(" in s:
        s = s[:s.rindex("(")].strip()
    return s


# Intersect with the known path set so any hallucinated paths silently drop.
known = set(all_files.keys())
selected = sorted({_strip_size_hint(s) for s in classified.selected} & known)
dropped = sorted(known - set(selected))

print(f"  Selected: {len(selected)} files")
print(f"  Dropped:  {len(dropped)} files")
if dropped:
    print(f"  Sample drops: {dropped[:10]}")

source = "\n\n".join(f"=== {p} ===\n{all_files[p]}" for p in selected)
print(f"Source blob: {len(source):,} chars")


# %% [cell 9 — host-side accumulator tool + RLM run]
from dspy.predict.rlm import RLM

# Reset on every run so re-executing this cell doesn't mix results
# from a previous run.  Dict preserves insertion order (Python 3.7+),
# so sections render in the order the LM first wrote them.
_sections: dict[str, str] = {}


def append_section(name: str, content: str) -> str:
    """Save one section of the contributor spec on the host.

    name: canonical section title (becomes the Markdown heading).
    content: Markdown body for that section.  May be multi-KB.
    Calling again with the same `name` overwrites previous content.
    """
    _sections[name] = content
    total = sum(len(v) for v in _sections.values())
    return (f"ok: saved '{name}' ({len(content):,} chars); "
            f"{len(_sections)} sections / {total:,} chars total.")


spec_instructions = SPEC_INSTRUCTIONS + "\n\n" + SECTION_PROTOCOL
spec_signature = SpecBuilderSignature.with_instructions(spec_instructions)

print("Running RLM (section-by-section delivery) ...")
predictor = RLM(spec_signature,
                max_iterations=MAX_ITERATIONS,
                max_llm_calls=MAX_LLM_CALLS,
                verbose=True,
                tools=[append_section])
result = predictor(repo_source=source)
print(f"\nSUBMIT status: {result.status!r}")
print(f"Sections collected: {len(_sections)}")
for k, v in _sections.items():
    print(f"  - {k}  ({len(v):,} chars)")


# %% [cell 10 — assemble + save + preview]
spec_text = "\n\n".join(f"## {name}\n\n{body}" for name, body in _sections.items())

if not spec_text.strip():
    print("WARNING: no sections captured.  Inspect `result.trajectory` "
          "to see what the LM actually did.")
else:
    pathlib.Path(OUT_PATH).write_text(spec_text, encoding="utf-8")
    print(f"Saved contributor spec to {OUT_PATH} "
          f"({len(spec_text):,} chars)\n")

    from IPython.display import Markdown, display
    display(Markdown(spec_text))


# %% [cell 11 — optional: download the spec]
# from google.colab import files
# files.download(OUT_PATH)
