"""
Chainer selection for the NL2PLN pipeline.

The pipeline is chainer-agnostic.  ``nl2pln.make_chainer`` is a zero-arg
factory returning an object that exposes the chainer interface used by
``difficulty_metric`` / ``eval_program.py``:

    add_atom(stmt)   # accumulate/assert a statement; raise on bad syntax
    query(term)      # run a query, return a stringifiable proof/result
    print_kb()       # debug dump

There is deliberately NO default backend: every entry point must select a
chainer explicitly (e.g. via a ``--chainer`` flag), so the pipeline never
silently assumes one.  ``resolve_chainer(spec)`` maps a short spec string
to a factory:

    pettachainer                      -> PeTTaChainer()            (../PeTTaChainer)
    lib_pln                           -> LibPlnChainer()           (lib_pln.metta in
                                         the installed PeTTa package)
    lib_pln:/path/to/lib_pln.metta    -> LibPlnChainer("/path/...")
    some.module:Factory               -> some.module.Factory()     (custom backend)

Backend imports are deferred into the factories, so importing this module
never pulls in a backend the caller did not choose (keeps ``--help`` cheap
and lets the pipeline run with only one backend installed).

The runtime contract a backend must satisfy is the ``Chainer`` protocol
defined below.  Adding a new chainer backend:

  1. Write an adapter class satisfying ``Chainer`` (``add_atom`` / ``query``
     / ``print_kb``), driving whatever the chainer actually is — a Python
     API, a MeTTa file on the PeTTa runtime, a subprocess, etc.
     ``chainers/lib_pln.py`` is a worked example.
  2. Make it reachable, either: add a one-line branch to ``resolve_chainer``
     for a friendly name, or use the generic ``module:Class`` spec with no
     edit here, e.g. ``--chainer my_pkg.my_module:MyChainer``.
  3. Run any entry point with ``--chainer <your-spec>``; the pipeline
     constructs one per puzzle -> add_atom the statements -> query -> score.
"""
import importlib
from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class Chainer(Protocol):
    """
    The runtime contract every chainer backend must satisfy.

    The pipeline creates a fresh instance per puzzle (via the zero-arg
    factory from ``resolve_chainer`` / ``nl2pln.make_chainer``), so each
    instance owns an isolated knowledge base.  It feeds the parser's PLN
    statements in with ``add_atom``, poses the queries with ``query``, and
    the query results become the "proof" the judge scores.

    This is a structural (duck-typed) Protocol: an adapter satisfies it
    just by having these methods — it need not subclass ``Chainer``.  Both
    PeTTaChainer (the package class) and ``LibPlnChainer`` already do.
    """

    def add_atom(self, statement: str) -> Any:
        """
        Assert one PLN statement into this instance's KB.

        MUST raise on a statement that is malformed for this chainer — that
        exception is the pipeline's syntax gate (a rejected statement makes
        the puzzle score as a parse failure).  The return value is ignored.
        """
        ...

    def query(self, query: str) -> Any:
        """
        Run one PLN query against the accumulated KB; return its proof.

        The result is stringified into the judge's prompt, so any
        stringifiable value is fine (e.g. a list of result strings, or an
        empty result when nothing is derivable).  Raise on a malformed
        query (same syntax-gate role as ``add_atom``).
        """
        ...

    def print_kb(self) -> None:
        """
        Print the current KB (debugging aid).  May be a no-op.
        """
        ...


def resolve_chainer(spec) -> "Callable[[], Chainer]":
    """
    Return a zero-arg factory producing a chainer for ``spec``.

    Raises ValueError for an unrecognized spec.  The factory itself defers
    the heavy backend import until it is called.
    """
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("chainer spec must be a non-empty string")
    spec = spec.strip()

    if spec == "pettachainer":
        def factory():
            from pettachainer import PeTTaChainer
            return PeTTaChainer()
        return factory

    if spec == "lib_pln" or spec.startswith("lib_pln:"):
        lib_path = spec.split(":", 1)[1] if ":" in spec else None
        def factory():
            from .lib_pln import LibPlnChainer
            return LibPlnChainer(lib_path)
        return factory

    if ":" in spec:
        # Generic escape hatch: "module.path:Callable" -> Callable()
        module_name, _, attr = spec.partition(":")
        def factory():
            module = importlib.import_module(module_name)
            return getattr(module, attr)()
        return factory

    raise ValueError(
        f"Unknown chainer spec {spec!r}.  Expected 'pettachainer', "
        f"'lib_pln[:<path-to-lib_pln.metta>]', or 'module.path:Callable'."
    )


def configure_chainer(spec):
    """
    Resolve ``spec`` and install it as ``nl2pln.make_chainer``.

    Returns the factory.  Call this from an entry point after parsing the
    ``--chainer`` flag and before running any program/metric.
    """
    import nl2pln
    factory = resolve_chainer(spec)
    nl2pln.make_chainer = factory
    return factory


def add_chainer_arg(parser, required=True):
    """
    Add the standard ``--chainer`` selection flag to an argparse parser.
    """
    parser.add_argument(
        "--chainer", required=required, default=None,
        help="Chainer backend to run against (REQUIRED; the pipeline has no "
             "default).  One of: 'pettachainer', "
             "'lib_pln[:<path-to-lib_pln.metta>]', or 'module.path:Callable'.",
    )
