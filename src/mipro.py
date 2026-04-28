"""
Optimize NL2PLNModule with MIPROv2, starting from bootstrap-generated content.

MIPROv2 is the heaviest non-SIMBA DSPy teleprompter but also the most
capable: it optimizes BOTH instructions AND demos jointly via Bayesian
optimization.  Crucially, unlike SIMBA's append-only rule mutation, the
proposer LLM can REWRITE, RESTRUCTURE, or CONDENSE the starting
instruction — full action space for fixing wrong or confusing parts of
the conversion guidelines.

Comparison with the other optimizer scripts in this repo:
  - bootstrapfewshot.py: demos only, instruction unchanged.  Lightest.
  - simba.py: SIMBA — appends rules, cannot delete or rewrite.  Heavier.
  - gepa_opt.py: GEPA — reflective rewriting, risk of drift.  Heaviest
    in iteration cost.
  - mipro.py (this): instruction rewrites + demos via Bayesian search.
    Good fit when the starting instruction has parts that need REMOVAL
    or REWRITING, not just additions.

Usage:
    python src/mipro.py
    python src/mipro.py --auto medium
    python src/mipro.py --auto heavy --max-bootstrapped-demos 6
    python src/mipro.py --no-auto --num-candidates 10 --num-trials 20
"""
import argparse
import os
import dspy
import json
import mlflow
import logging
from pprint import pformat
from dspy.teleprompt import MIPROv2
from dspy.utils.callback import BaseCallback
from pathlib import Path

from nl2pln import NL2PLNModule , difficulty_metric , build_examples_from_file
from logging_utils import setup_logging

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  LM choices previously tried (pass any LiteLLM id via --model or             #
#  --prompt-model):                                                            #
#   openai/gpt-5.4-mini                                                       #
#   openai/gpt-5.4                                                            #
#   openai/gpt-5.2                                                            #
#   openrouter/openai/gpt-5.1                                                 #
#   openrouter/deepseek/deepseek-v3.2                                         #
#   cerebras/gpt-oss-120b                                                     #
#   openrouter/google/gemini-3-pro-preview                                    #
#   openrouter/google/gemini-3-flash-preview                                  #
#   openrouter/z-ai/glm-4.5                                                   #
#   moonshotai/kimi-k2-0905:exacto                                            #
# --------------------------------------------------------------------------- #


def parse_args():
    p = argparse.ArgumentParser(
        description="Run MIPROv2 optimization on NL2PLNModule (instruction + demos)"
    )
    # ---- LM config ----
    p.add_argument("--model", default="openai/gpt-5.4-mini",
                   help="LiteLLM model id for the TASK LM (used during eval trajectories)")
    p.add_argument("--prompt-model", default=None,
                   help="LiteLLM model id for the PROPOSER LM (called few times to "
                        "generate candidate instructions).  Recommended: a stronger "
                        "model than --model (e.g. openai/gpt-5.4).  If unset, uses --model.")
    p.add_argument("--reasoning-effort", default="high",
                   help="Reasoning effort for both LMs: none, low, medium, high, xhigh")
    # ---- data ----
    p.add_argument("--dataset", default="data/generated.json",
                   help="Training dataset path")
    p.add_argument("--valset", default=None,
                   help="Optional separate validation set path.  If unset, MIPROv2 "
                        "splits the trainset internally.")
    # ---- MIPROv2 parameters ----
    p.add_argument("--auto", default="light",
                   choices=["light", "medium", "heavy", "none"],
                   help="Auto-budget preset.  'light'=~5 candidates, 'medium'=~10, "
                        "'heavy'=~20.  Use 'none' to manually set --num-candidates + "
                        "--num-trials.")
    p.add_argument("--num-candidates", type=int, default=None,
                   help="[--auto none only] How many instruction/demo candidates to generate")
    p.add_argument("--num-trials", type=int, default=None,
                   help="[--auto none only] How many Bayesian-search trials")
    p.add_argument("--max-bootstrapped-demos", type=int, default=4,
                   help="Max bootstrapped demos per predictor")
    p.add_argument("--max-labeled-demos", type=int, default=0,
                   help="Max labeled demos (typically 0 — our trainset has no PLN labels)")
    p.add_argument("--num-threads", type=int, default=10,
                   help="Parallel LM calls during evaluation")
    p.add_argument("--seed", type=int, default=21,
                   help="Random seed for trainset shuffle and MIPROv2 reproducibility "
                        "(matches simba.py / bootstrapfewshot.py / gepa_opt.py)")
    # ---- starting content ----
    p.add_argument("--instruction-file", default="instructions.md",
                   help="File whose contents become the starting signature instruction "
                        "(empty string to use the baseline NL2PLNSignature instruction)")
    p.add_argument("--pln-spec-file", default="chainer_analysis.txt",
                   help="File whose contents become the pln_spec input value "
                        "(empty string to use PeTTaChainer's default LLM_RULE_SPEC.md)")
    p.add_argument("--input", default=None,
                   help="Optional checkpoint to seed the module before optimization")
    p.add_argument("--output", default="programs/mipro.json",
                   help="Where to save the optimized program")
    # ---- logging ----
    p.add_argument("--log-dir", default=None,
                   help="Optional directory for MIPROv2 run logs")
    p.add_argument("--log-level", default="info",
                   choices=["debug", "info", "warning", "error"],
                   help="Logging verbosity (default: info)")
    p.add_argument("--log-file", default="/tmp/mipro.log",
                   help="Path to log file; mirrors stdout/stderr for Colab "
                        "where streaming cell output gets truncated.")
    return p.parse_args()


def main():
    args = parse_args()

    setup_logging(args.log_file, args.log_level)
    print(f"  Logging to {args.log_file} (level={args.log_level})")

    lm_kwargs = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else {}
    task_lm = dspy.LM(args.model, **lm_kwargs)
    prompt_lm = dspy.LM(args.prompt_model, **lm_kwargs) if args.prompt_model else task_lm
    # Configure task_lm as the default so NL2PLNModule.forward() uses it
    dspy.configure(lm=task_lm)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(uri=tracking_uri)
        mlflow.set_experiment("DSPy-Optimization")
        mlflow.dspy.autolog(
            log_compiles=True,    # Track optimization process
            log_evals=True,       # Track evaluation results
            log_traces_from_compile=True  # Track program traces during optimization
        )

    # Override pln_spec from file if requested. NL2PLNModule.forward() reads
    # `pln_spec` from the module-level constant in nl2pln.py, so monkey-patching
    # that attribute changes what every subsequent .forward() call sees.
    if args.pln_spec_file:
        pln_spec_path = Path(args.pln_spec_file)
        if pln_spec_path.exists():
            new_spec = pln_spec_path.read_text(encoding="utf-8")
            import nl2pln
            nl2pln.pln_spec = new_spec
            print(f"  Overrode pln_spec from {pln_spec_path} ({len(new_spec):,} chars)")
        else:
            logger.warning("pln_spec file %s not found; using default", pln_spec_path)

    trainset = build_examples_from_file(args.dataset)
    print(f"  Loaded {len(trainset)} training examples from {args.dataset}")

    # Shuffle so MIPROv2's bootstrap-demo phase (which iterates the trainset
    # in order) sees phenomenon-diverse samples instead of clustering on the
    # first few phenomena from `data/generated.json`.
    import random
    random.Random(args.seed).shuffle(trainset)

    valset = None
    if args.valset:
        valset = build_examples_from_file(args.valset)
        print(f"  Loaded {len(valset)} validation examples from {args.valset}")

    module = NL2PLNModule()
    if args.input:
        input_path = Path(args.input)
        if input_path.exists():
            module.load(str(input_path))
            print(f"  Loaded checkpoint from {input_path}")
        else:
            logger.info("No checkpoint found at %s; starting from uninitialized module.", input_path)

    # Override the signature instruction from file if requested. DSPy 3.2's
    # ChainOfThought doesn't expose .signature directly; iterate via
    # named_predictors() to reach the underlying Predict objects that do.
    if args.instruction_file:
        instr_path = Path(args.instruction_file)
        if instr_path.exists():
            new_instruction = instr_path.read_text(encoding="utf-8")
            for _, predictor in module.named_predictors():
                predictor.signature = predictor.signature.with_instructions(new_instruction)
            print(f"  Overrode instruction from {instr_path} ({len(new_instruction):,} chars)")
        else:
            logger.warning("instruction file %s not found; using default", instr_path)

    # MIPROv2's `auto` preset and manual (num_candidates, num_trials) are
    # mutually exclusive.  Translate --auto none to auto=None.
    auto_value = None if args.auto == "none" else args.auto
    if auto_value is None and (args.num_candidates is None or args.num_trials is None):
        raise SystemExit(
            "When --auto none is used, both --num-candidates and --num-trials must be set."
        )

    mipro_kwargs = {
        "metric": difficulty_metric,
        "prompt_model": prompt_lm,
        "task_model": task_lm,
        "max_bootstrapped_demos": args.max_bootstrapped_demos,
        "max_labeled_demos": args.max_labeled_demos,
        "num_threads": args.num_threads,
        "seed": args.seed,
        "auto": auto_value,
        "verbose": True,
        "track_stats": True,
        "log_dir": args.log_dir,
    }
    if auto_value is None:
        mipro_kwargs["num_candidates"] = args.num_candidates

    teleprompter = MIPROv2(**mipro_kwargs)

    compile_kwargs = {
        "student": module,
        "trainset": trainset,
    }
    if valset is not None:
        compile_kwargs["valset"] = valset
    if auto_value is None:
        compile_kwargs["num_trials"] = args.num_trials

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Wrap compile() so that an uncaught exception still leaves a saved
    # snapshot.  This is the protection that was missing on the prior
    # MIPROv2 run that crashed post-save — that run got lucky (the save
    # had already happened); a crash a few seconds earlier would have
    # lost everything.
    compile_succeeded = False
    try:
        module = teleprompter.compile(**compile_kwargs)
        compile_succeeded = True
    finally:
        try:
            module.save(str(output_path))
            if compile_succeeded:
                print(f"  Saved optimized program to {output_path}")
            else:
                print(f"  CRASH-SAVE: wrote partial/pre-crash module state to {output_path}")
        except Exception as save_err:
            # Don't mask the original exception; just report save failure
            print(f"  CRASH-SAVE FAILED: {save_err}")


if __name__ == "__main__":
    main()
