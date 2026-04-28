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
from pettachainer import PeTTaChainer, get_language_spec

pln_spec = get_language_spec(llm_focused=True)

class NL2PLNSingature(dspy.Signature):
    """Convert natural language to PLN light statements and queries.

    Follow `pln_spec` exactly and reuse predicates from `context` when possible.
    """
    #Inputs
    sentences: List[str] = dspy.InputField(desc="Original natural language sentences")
    context: List[str] = dspy.InputField(desc="Contextual information")
    pln_spec: str = dspy.InputField(desc="PLN light syntax and semantics specification")

    #Outputs
    statements: List[str] = dspy.OutputField(desc="PLN light statements to add to the knowledge base")
    queries: List[str] = dspy.OutputField(desc="PLN light queries for question answering")

class NL2PLNModule(dspy.Module):

    def __init__(self):
        self.nl2pln : dspy.Module = dspy.ChainOfThought(NL2PLNSingature)

    def forward(self, sentences : List[str], queries: List[dict]):
        base = self.nl2pln(sentences=sentences, context=[], pln_spec=pln_spec)
        stmts = [] if base.statements is None else list(base.statements)
        seen = set(stmts)
        context_stmts = list(stmts)

        queries_pln = []
        for q in queries:
            pln_q = self.nl2pln(sentences=[q['question']], context=context_stmts, pln_spec=pln_spec)
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
    """Evaluate how well a proof answers a question and suggest improvements.

    You are evaluating PLN (Probabilistic Logic Networks) proofs generated from natural language.
    Assess whether the proof correctly answers the question and provide constructive feedback.
    """
    # Inputs
    sentences: List[str] = dspy.InputField(desc="Original natural language sentences")
    question: str = dspy.InputField(desc="The question being asked")
    expected_answer: str = dspy.InputField(desc="The expected answer to the question")
    pln_spec: str = dspy.InputField(desc="Current PLN light syntax and semantics specification")
    statements: List[str] = dspy.InputField(desc="PLN statements generated from sentences")
    query: List[str] = dspy.InputField(desc="PLN query generated for the question")
    proof: str = dspy.InputField(desc="The proof result from running the query")

    # Outputs
    score: float = dspy.OutputField(desc="Score from 0.0 to 1.0 indicating how well the proof answers the question")
    feedback: str = dspy.OutputField(desc="Detailed feedback on what went wrong and how to improve the PLN statements/query")
    improved_statements: List[str] = dspy.OutputField(desc="Improved PLN statements that would produce a better proof")
    improved_query: List[str] = dspy.OutputField(desc="Improved PLN query that would better capture the question")


class ProofEvaluator(dspy.Module):
    def __init__(self):
        self.evaluate = dspy.ChainOfThought(ProofEvaluatorSignature)

    def forward(self, sentences, question, expected_answer, pln_spec, statements, query, proof):
        return self.evaluate(
            sentences=sentences,
            question=question,
            expected_answer=expected_answer,
            pln_spec=pln_spec,
            statements=statements,
            query=query,
            proof=proof
        )

def difficulty_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None, pred_name=None, pred_trace=None):
    # Defensive guard: when the upstream LM call fails (auth/quota/rate-
    # limit/network), DSPy passes pred=None into the metric.  Without this
    # guard, the broad except below would catch the AttributeError on
    # `pred.statements` and silently return score=0.0 — the failure mode
    # that lost three optimizer runs to date (SIMBA spinning through
    # all-zero batches with the crash-save never firing).  Fail fast so
    # the optimizer's try/finally has a chance to preserve in-progress
    # state.
    if pred is None:
        raise RuntimeError(
            "difficulty_metric received pred=None — upstream LM call failed "
            "(likely auth, quota, rate-limit, or network).  Failing fast so "
            "the optimizer's crash-save can preserve in-progress state."
        )

    try:
        metta_handler = PeTTaChainer()
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

        # Inner query lists are conjunctive (witness sets for universal/cardinal
        # claims), so run all queries per question and collect all proofs.
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
            evaluation = evaluator(
                sentences=gold.sentences,
                question=q['question'],
                expected_answer=q['expected_answer'],
                pln_spec=pln_spec,
                statements=pred.statements,
                query=query_pln,
                proof=formatted_proof
            )

            eval_score = 0.0 if evaluation.score is None else float(evaluation.score)
            total_score += eval_score

            feedback_details.append(dedent(f"""
                Question: '{q['question']}'
                Expected: {q['expected_answer']}
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
        # Re-raise LM-side errors so the optimizer fails fast and the
        # crash-save in simba.py / mipro.py / etc. fires.  Genuine
        # semantic failures (PeTTaChainer parse errors, missing fields,
        # etc.) continue to score 0.0 below as before.  We match by
        # class name so we don't have to import litellm here.
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
