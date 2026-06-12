"""
LibPlnChainer — adapter exposing the NL2PLN chainer interface over the
``lib_pln.metta`` PLN library that ships inside the PeTTa runtime.

Why this adapter is non-trivial (vs. PeTTaChainer):

  * No Python API and no persistent KB.  Knowledge is passed *as an
    argument* to ``(PLN.Query <kb> <term>)``, where ``<kb>`` is a tuple of
    ``(Sentence (<term> (stv s c)) <evidence-stamp>)`` entries.  So this
    adapter accumulates asserted statements in Python and assembles the KB
    tuple at query time.  (A nice side effect: each instance's KB lives in
    Python, so instances are naturally isolated even though the underlying
    PeTTa runtime is a shared global.)

  * Evidence stamps.  Derivations combine only evidence-disjoint inputs, so
    each asserted statement gets a distinct stamp ``(1)``, ``(2)``, ...  The
    translator emits a bare light statement ``(<term> (stv s c))``; this
    adapter wraps it in the Sentence envelope and assigns the stamp.  A full
    ``(Sentence ...)`` envelope is also accepted and re-stamped.

  * Only ``(stv strength confidence)`` truth values exist (no distributions).

Marginal-STV caveat: several lib_pln rules (Inheritance/Implication
deduction, induction, abduction, inversion, equivalence->implication,
transitive similarity, evaluation-implication, member-deduction) only fire
when marginal ``(= (STV <term>) (stv s c))`` *program equations* are present
(the default returns empty).  This minimal adapter does NOT inject
marginals, so only the marginal-free fragment (modus ponens,
Not-elimination, the predicate-specialization rules) derives results.  Add
marginal support later if the smoke test shows deduction-heavy puzzles
returning empty proofs.
"""
from pathlib import Path


# ---------------------------------------------------------------------------
# Minimal S-expression parser — enough to validate shape and (un)wrap terms.
# ---------------------------------------------------------------------------

def _tokenize(s):
    return s.replace("(", " ( ").replace(")", " ) ").split()


def _parse(s):
    """
    Parse exactly one S-expression into nested lists of str.

    Raises ValueError on empty input, unbalanced parens, or trailing tokens
    after the first expression.  This doubles as the adapter's syntax gate.
    """
    tokens = _tokenize(s)
    if not tokens:
        raise ValueError("empty expression")
    pos = 0

    def parse_one():
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok == "(":
            items = []
            while pos < len(tokens) and tokens[pos] != ")":
                items.append(parse_one())
            if pos >= len(tokens):
                raise ValueError("unbalanced '('")
            pos += 1  # consume ')'
            return items
        if tok == ")":
            raise ValueError("unexpected ')'")
        return tok

    tree = parse_one()
    if pos != len(tokens):
        raise ValueError("more than one top-level expression")
    return tree


def _ser(node):
    if isinstance(node, list):
        return "(" + " ".join(_ser(x) for x in node) + ")"
    return node


def _is_stv(node):
    return isinstance(node, list) and len(node) == 3 and node[0] == "stv"


def _validate_stv(node):
    for x in (node[1], node[2]):
        try:
            float(x)
        except (TypeError, ValueError):
            raise ValueError(f"stv strength/confidence must be numeric, got {x!r}")


def _contains_var(node):
    if isinstance(node, list):
        return any(_contains_var(x) for x in node)
    return isinstance(node, str) and node.startswith("$")


def _split_term_tv(tree):
    """
    From a light statement ``(<term> (stv s c))`` or a full
    ``(Sentence (<term> (stv s c)) <stamp>)`` envelope, return
    ``(term_node, tv_node)``.  Raises ValueError otherwise.
    """
    if isinstance(tree, list) and tree and tree[0] == "Sentence":
        if len(tree) < 2 or not isinstance(tree[1], list) or len(tree[1]) != 2:
            raise ValueError("malformed Sentence envelope")
        inner = tree[1]
        if not _is_stv(inner[1]):
            raise ValueError("Sentence inner pair must be (<term> (stv s c))")
        return inner[0], inner[1]
    if isinstance(tree, list) and len(tree) == 2 and _is_stv(tree[1]):
        return tree[0], tree[1]
    raise ValueError(
        "expected '(<term> (stv s c))' or "
        "'(Sentence (<term> (stv s c)) <stamp>)'"
    )


def _default_lib_path():
    try:
        import petta
    except ImportError as e:
        raise ImportError(
            "the 'petta' package is required for the lib_pln chainer"
        ) from e
    root = Path(petta.__file__).resolve().parent.parent
    candidate = root / "lib" / "lib_pln.metta"
    if not candidate.exists():
        raise FileNotFoundError(
            f"could not locate lib_pln.metta at {candidate}; pass an explicit "
            f"path via the 'lib_pln:<path>' chainer spec"
        )
    return str(candidate)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class LibPlnChainer:
    # lib_pln.metta defines global rules; load each lib file once per process.
    _loaded_libs = set()

    def __init__(self, lib_path=None):
        from petta import PeTTa
        self.handler = PeTTa()
        self.lib_path = str(lib_path) if lib_path else _default_lib_path()
        if self.lib_path not in LibPlnChainer._loaded_libs:
            self.handler.load_metta_file(self.lib_path)
            LibPlnChainer._loaded_libs.add(self.lib_path)
        self._sentences = []   # ["(Sentence (<term> (stv s c)) (<id>))", ...]
        self._next_stamp = 1

    def add_atom(self, atom):
        """
        Accumulate a statement into the KB.  Raises ValueError on bad
        syntax (the pipeline's syntax gate) and returns the wrapped Sentence.
        """
        tree = _parse(atom)
        if _contains_var(tree):
            raise ValueError(
                f"external assertions must be ground; '$' variable in {atom!r}"
            )
        term, tv = _split_term_tv(tree)
        _validate_stv(tv)
        stamp = f"({self._next_stamp})"
        self._next_stamp += 1
        sentence = f"(Sentence ({_ser(term)} {_ser(tv)}) {stamp})"
        self._sentences.append(sentence)
        return sentence

    def query(self, atom, maxsteps=None):
        """
        Run a query against the accumulated KB.  ``atom`` is the bare
        target term; a wrapped statement/Sentence is tolerated and reduced
        to its term.  Returns the raw PLN.Query result list.
        """
        tree = _parse(atom)
        if isinstance(tree, list) and tree and tree[0] == "Sentence":
            tree = _split_term_tv(tree)[0]
        elif isinstance(tree, list) and len(tree) == 2 and _is_stv(tree[1]):
            tree = tree[0]
        if _contains_var(tree):
            raise ValueError(
                f"lib_pln queries must be ground terms; '$' variable in {atom!r}"
            )
        kb = "(" + " ".join(self._sentences) + ")"
        steps = "" if maxsteps is None else f" {maxsteps}"
        return self.handler.process_metta_string(
            f"!(PLN.Query {kb} {_ser(tree)}{steps})"
        )

    def print_kb(self):
        for s in self._sentences:
            print(s)
