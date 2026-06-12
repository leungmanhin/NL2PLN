import os
import dspy
import json
import logging
import mlflow
import traceback
from pathlib import Path

logger = logging.getLogger(__name__)

from typing import List
from textwrap import dedent

# Module-level pln_spec.  Defaults to empty; callers (optimizer scripts,
# eval_program.py, usage_example.py, etc.) set this from a chainer-
# specific spec file (e.g. bootstrap/chainer_analysis.txt produced by
# bootstrap_chainer.py) before constructing NL2PLNModule.  The content
# is injected into the signature instruction (system-prompt slot) at
# module construction and after every load(), so the LM still sees the
# spec but it's rendered once per call rather than once per demo —
# whereas storing pln_spec per-call as an InputField bloated prompts
# at inference (e.g. 6 demos × 28KB = 173KB of duplicated content per
# call).
pln_spec = ""

# Markers bracketing the auto-injected pln_spec section in the
# signature instruction.  Used to make injection idempotent (re-
# injection strips any prior section before adding a fresh one) and
# to leave any user-provided instruction text around the section
# untouched.
_PLN_SPEC_BEGIN = "<!-- BEGIN_PLN_SPEC -->"
_PLN_SPEC_END = "<!-- END_PLN_SPEC -->"

# Chainer factory.  The pipeline is chainer-agnostic and has NO default
# backend: callers MUST install a zero-arg factory here before running
# difficulty_metric (e.g. via chainers.configure_chainer("lib_pln") or a
# script's --chainer flag).  The factory returns an object exposing
# add_atom(stmt) / query(q) / print_kb().  Left None so that forgetting to
# select a chainer fails loudly rather than silently assuming one.
make_chainer = None


class NL2PLNSingature(dspy.Signature):
    """
    Convert natural language to PLN light statements and queries.

    Follow the PLN spec described at the top of these instructions
    and reuse predicates from `context` when possible.
    """
    #Inputs
    sentences: List[str] = dspy.InputField(desc="Original natural language sentences")
    context: List[str] = dspy.InputField(desc="Contextual information")

    #Outputs
    statements: List[str] = dspy.OutputField(desc="PLN light statements to add to the knowledge base")
    queries: List[str] = dspy.OutputField(desc="PLN light queries for question answering")


class NL2PLNModule(dspy.Module):

    def __init__(self):
        self.nl2pln: dspy.Module = dspy.ChainOfThought(NL2PLNSingature)
        self._inject_pln_spec()

    def _inject_pln_spec(self):
        """
        Inject the current module-level ``pln_spec`` content into the
        signature instructions of every predictor.  Idempotent: strips
        any prior auto-injected section first, then re-adds.  Also
        strips legacy ``pln_spec`` keys from any loaded demos
        (backward-compat cleanup for programs trained when pln_spec was
        an InputField — those demos still carry full pln_spec snapshots
        in their dicts).
        """
        for _, predictor in self.named_predictors():
            instr = predictor.signature.instructions
            instr = self._strip_pln_spec_section(instr)
            if pln_spec:
                instr = (
                    f"{_PLN_SPEC_BEGIN}\n"
                    f"## PLN spec\n\n{pln_spec}\n"
                    f"{_PLN_SPEC_END}\n\n"
                    f"{instr}"
                )
            predictor.signature = predictor.signature.with_instructions(instr)
            self._strip_pln_spec_from_demos(predictor)

    @staticmethod
    def _strip_pln_spec_section(instr: str) -> str:
        """
        Remove any prior auto-injected pln_spec section, leaving the
        rest of the instruction untouched.  No-op if the markers aren't
        present.
        """
        begin = instr.find(_PLN_SPEC_BEGIN)
        if begin == -1:
            return instr
        end = instr.find(_PLN_SPEC_END, begin)
        if end == -1:
            return instr
        end_after = end + len(_PLN_SPEC_END)
        before = instr[:begin]
        after = instr[end_after:].lstrip("\n")
        return before + after

    @staticmethod
    def _strip_pln_spec_from_demos(predictor) -> None:
        """
        Remove the legacy ``pln_spec`` key from saved demos.  Old
        programs (pre-this-refactor) saved pln_spec as an InputField,
        so each demo dict carries a full copy.  Removing pln_spec from
        the signature stops it being rendered into prompts, but the
        demo dicts still hold the data — strip it for cleanliness so
        re-saving the program writes out lighter JSON too.
        """
        for demo in predictor.demos:
            try:
                if hasattr(demo, "_store") and "pln_spec" in demo._store:
                    del demo._store["pln_spec"]
                elif isinstance(demo, dict) and "pln_spec" in demo:
                    del demo["pln_spec"]
            except Exception:
                pass

    def load(self, path):
        """
        Load saved program state, then re-inject pln_spec.

        DSPy's load_state restores the signature's instructions from the
        saved JSON.  Old programs' saved instructions don't contain a
        pln_spec section (it was a separate InputField), and even
        new-format programs saved with one pln_spec may now want a
        different one applied (e.g. user changed the global before
        loading).  Re-injecting after load handles both cases.
        """
        super().load(path)
        self._inject_pln_spec()

    def forward(self, sentences: List[str], queries: List[dict]):
        # pln_spec is no longer passed per-call — it's in the signature
        # instructions, applied automatically on every LM call.
        base = self.nl2pln(sentences=sentences, context=[])
        stmts = [] if base.statements is None else list(base.statements)
        seen = set(stmts)
        context_stmts = list(stmts)

        queries_pln = []
        for q in queries:
            pln_q = self.nl2pln(sentences=[q['question']], context=context_stmts)
            q_stmts = [] if pln_q.statements is None else list(pln_q.statements)
            for s in q_stmts:
                if s not in seen:
                    seen.add(s)
                    stmts.append(s)
                    context_stmts.append(s)
            q_queries = [] if pln_q.queries is None else list(pln_q.queries)
            queries_pln.append(q_queries)

        return dspy.Prediction(statements=stmts, queries=queries_pln)

class ProofEvaluatorSignature(dspy.Signature):
    """
    Evaluate how well a proof answers a question and suggest improvements.

    You are evaluating PLN (Probabilistic Logic Networks) proofs generated from natural language.
    Assess whether the proof correctly answers the question and provide constructive feedback.

    If `constraints` is non-empty, the proof must additionally satisfy each
    listed constraint (e.g. "STV strength in [0.3, 0.7]").  Constraints are
    short, self-contained, machine-checkable claims about the proof's content;
    verify each against the actual proof string.  A proof that answers the
    question but violates the constraints should receive a reduced score
    (roughly proportional to how many constraints are unsatisfied and how
    central they are to the question's intent).  If `constraints` is empty,
    score on expected_answer match alone.
    """
    # Inputs
    sentences: List[str] = dspy.InputField(desc="Original natural language sentences")
    question: str = dspy.InputField(desc="The question being asked")
    expected_answer: str = dspy.InputField(desc="The expected answer to the question")
    pln_spec: str = dspy.InputField(desc="Current PLN light syntax and semantics specification")
    statements: List[str] = dspy.InputField(desc="PLN statements generated from sentences")
    query: List[str] = dspy.InputField(desc="PLN query generated for the question")
    proof: str = dspy.InputField(desc="The proof result from running the query")
    constraints: List[str] = dspy.InputField(
        desc='Optional list of additional constraints the proof should satisfy '
             '(e.g. "STV strength in [0.3, 0.7]").  Empty list means no '
             'additional constraints — score on expected_answer match alone.'
    )

    # Outputs
    score: float = dspy.OutputField(desc="Score from 0.0 to 1.0 indicating how well the proof answers the question")
    feedback: str = dspy.OutputField(desc="Detailed feedback on what went wrong and how to improve the PLN statements/query")
    improved_statements: List[str] = dspy.OutputField(desc="Improved PLN statements that would produce a better proof")
    improved_query: List[str] = dspy.OutputField(desc="Improved PLN query that would better capture the question")


class ProofEvaluator(dspy.Module):
    def __init__(self):
        self.evaluate = dspy.ChainOfThought(ProofEvaluatorSignature)

    def forward(self, sentences, question, expected_answer, pln_spec, statements, query, proof, constraints):
        return self.evaluate(
            sentences=sentences,
            question=question,
            expected_answer=expected_answer,
            pln_spec=pln_spec,
            statements=statements,
            query=query,
            proof=proof,
            constraints=constraints,
        )

def difficulty_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None, pred_name=None, pred_trace=None):
    # Fail fast on pred=None (upstream LM failure) — otherwise the broad except below silently returns 0.0.
    if pred is None:
        raise RuntimeError(
            "difficulty_metric received pred=None — upstream LM call failed "
            "(likely auth, quota, rate-limit, or network)."
        )

    if make_chainer is None:
        raise RuntimeError(
            "No chainer configured: set nl2pln.make_chainer to a zero-arg "
            "factory (e.g. via chainers.configure_chainer('lib_pln') or a "
            "script's --chainer flag) before running difficulty_metric."
        )
    # Construct outside the try below so a missing/broken backend fails loudly
    # rather than being swallowed into a per-example score of 0.0.
    metta_handler = make_chainer()

    try:
        evaluator = ProofEvaluator()

        log = False

        score = 0.0
        if pred.statements == [] or pred.statements is None:
            return dspy.Prediction(score=score, feedback="No pln statements found")

        for stmt in pred.statements:
            try:
                metta_handler.add_atom(stmt)
            except Exception as e:
                return dspy.Prediction(
                    score=score,
                    feedback=f"""The statement {stmt} did not follow the right syntax. Follow the pln light spec {pln_spec}. Details: {e}"""
                )
            score += 0.001

        metta_handler.print_kb()

        # Inner query lists are conjunctive (witness sets for universal/cardinal claims), so run all per question.
        proofs = []
        for qr in pred.queries:
            qr_proofs = []
            for q in qr:
                try:
                    qr_proofs.append(metta_handler.query(q))
                except Exception as e:
                    return dspy.Prediction(
                        score=score,
                        feedback=f"""The query {q} did not follow the right syntax. Follow the pln light spec {pln_spec}. Details: {e}"""
                    )
                score += 0.001
            proofs.append(qr_proofs)

        total_score = 0.0
        feedback_details = []

        for q, query_pln, proof_list in zip(gold.queries, pred.queries, proofs):
            formatted_proof = "\n".join(
                f"Query: {qry}\nProof: {p}"
                for qry, p in zip(query_pln, proof_list)
            )
            constraints = q.get('constraints') or []
            evaluation = evaluator(
                sentences=gold.sentences,
                question=q['question'],
                expected_answer=q['expected_answer'],
                pln_spec=pln_spec,
                statements=pred.statements,
                query=query_pln,
                proof=formatted_proof,
                constraints=constraints,
            )

            eval_score = 0.0 if evaluation.score is None else float(evaluation.score)
            total_score += eval_score

            feedback_details.append(dedent(f"""
                Question: '{q['question']}'
                Expected: {q['expected_answer']}
                Constraints: {constraints}
                Proofs:
{formatted_proof}
                Score: {eval_score}
                Feedback: {evaluation.feedback}
                Improved statements: {evaluation.improved_statements}
                Improved query: {evaluation.improved_query}
            """))

        n = len(pred.queries)
        final_score = max(total_score / n if n > 0 else 0.0, 0.1)

        return dspy.Prediction(
            score=final_score,
            feedback=f"Score: {total_score:.2f}/{n} questions. \n" + "\n".join(feedback_details)
        )
    except Exception as e:
        # Re-raise LM-side errors (matched by class name to avoid importing litellm); semantic failures still score 0.0.
        if type(e).__name__ in (
            "RateLimitError", "AuthenticationError",
            "APIConnectionError", "BadRequestError",
            "ServiceUnavailableError", "APIError",
            "Timeout", "InternalServerError",
        ):
            raise
        print(pred)
        print("Error occured in difficulty_metric:", e)
        traceback.print_exc()
        return dspy.Prediction(score=0.0, feedback=f"Error: {e}")



def build_examples_from_file(filepath: str) -> List[dspy.Example]:
    """
    Load a JSON file containing a list of puzzle data and convert each item into a
    dspy.Example that provides 'sentences' (list[str]) and 'questions' (list[dict]) as inputs.
    """
    examples: List[dspy.Example] = []
    with open(filepath, "r", encoding="utf-8") as f:
        puzzle_data = json.load(f)
    for item in puzzle_data:
        examples.append(
            dspy.Example(item).with_inputs("sentences", "queries")
        )
    return examples

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(uri=tracking_uri)
    mlflow.set_experiment("DSPy-Optimization")
    mlflow.dspy.autolog(
        log_compiles=True,    # Track optimization process
        log_evals=True,       # Track evaluation results
        log_traces_from_compile=True  # Track program traces during optimization
    )

if __name__ == '__main__':
    #model = "openrouter/openai/gpt-5.1"
    #model = "openrouter/deepseek/deepseek-v3.2"
    #model = "cerebras/gpt-oss-120b"
    #model = "openrouter/google/gemini-3-flash-preview"
    model = "openai/gpt-5.2"

    dspy.configure(
        lm=dspy.LM(model,temperature=1.0, max_tokens=20000),
        #enable_disk_cache=False,
        #enable_memory_cache=False,
    )
    dspy.settings.configure(track_usage=True)

    # Dev harness: select a chainer explicitly (the pipeline has no default).
    import chainers
    make_chainer = chainers.resolve_chainer("pettachainer")

    module = NL2PLNModule()
    #compiled_program = Path("programs/simba_all3.json")
    #if compiled_program.exists():
    #    module.load(str(compiled_program))
    #else:
    #    logger.info("No compiled program found at %s; running module without load().", compiled_program)

    puzzle_data = build_examples_from_file("data/all.json")
    puzzle_data = puzzle_data[0:1]

    score_sum = 0
    for puzzle in puzzle_data:
        res = module(sentences=puzzle.sentences, queries=puzzle.queries)
        print("------------------------------------------------------------------------------------------------------------------------------")
        print(res)
        metric = difficulty_metric(puzzle, res)
        print("------------------------------------------------------------------------------------------------------------------------------")
        print(metric.score)
        print(metric.feedback)
        score_sum += metric.score
    print(score_sum/len(puzzle_data))
