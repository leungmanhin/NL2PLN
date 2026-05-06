"""
Optimize NL2PLNModule with GEPA — reflective evolution of the signature
instruction.

GEPA is the heaviest of the optimizers in this repo: a reflection LM rewrites
the instruction across many iterations based on what worked / didn't.  Higher
ceiling than SIMBA's append-only mutation, but also higher drift risk and
much higher cost (num_metric_calls dominates).  Typical workflow is
GEPA-on-top-of-SIMBA: load a SIMBA-trained checkpoint via --input and let
GEPA refine the (already-evolved) instruction further.

Comparison with the other optimizer scripts in this repo:
  - bootstrapfewshot.py: demos only, instruction unchanged.  Lightest.
  - simba.py:   SIMBA — appends rules, cannot delete or rewrite.  Heavier.
  - mipro.py:   MIPROv2 — instruction rewrites + demos via Bayesian search.
  - gepa_opt.py (this): reflective rewriting, highest iteration cost.

Usage:
    python src/gepa_opt.py
    python src/gepa_opt.py --input programs/simba.json
    python src/gepa_opt.py --max-metric-calls 500 --score-threshold 0.85
"""
import argparse
import os
import dspy
import json
import logging
import mlflow
from pprint import pformat
from gepa.utils.stop_condition import MaxMetricCallsStopper, ScoreThresholdStopper
from dspy.teleprompt import GEPA
from dspy.utils.callback import BaseCallback
from pathlib import Path

from nl2pln import NL2PLNModule , difficulty_metric , build_examples_from_file
from logging_utils import setup_logging

# GEPA's internal logger — preserved from the original script so its
# reflection-step logs continue to flow through this configuration.
logger = logging.getLogger(dspy.teleprompt.gepa.gepa.__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run GEPA optimization on NL2PLNModule (typically bootstraps from a SIMBA checkpoint)"
    )
    # ---- LM config ----
    p.add_argument("--model", default="openai/gpt-5.4-mini",
                   help="LiteLLM model id used for the task LM")
    p.add_argument("--reflection-model", default=None,
                   help="LiteLLM model id for the REFLECTION LM (called to propose "
                        "instruction rewrites).  Recommended: a stronger model than "
                        "--model.  If unset, uses --model.")
    p.add_argument("--reasoning-effort", default="high",
                   help="Reasoning effort for reasoning-capable models: none, low, medium, high, xhigh")
    # ---- data ----
    p.add_argument("--dataset", default="data/generated.json",
                   help="Training dataset path (default: bootstrap-generated training data)")
    p.add_argument("--valset", default=None,
                   help="Optional separate validation set path.  If unset, the "
                        "trainset is reused as the valset (preserves the original "
                        "behavior of this script).")
    # ---- GEPA parameters ----
    p.add_argument("--num-threads", type=int, default=10,
                   help="Parallel LM calls during optimization")
    p.add_argument("--max-metric-calls", type=int, default=1000,
                   help="GEPA evaluation budget (total metric calls)")
    p.add_argument("--reflection-minibatch-size", type=int, default=16,
                   help="Minibatch size for GEPA reflective mutation")
    p.add_argument("--score-threshold", type=float, default=0.9,
                   help="Early-stop threshold: stop when a candidate scores >= this")
    p.add_argument("--seed", type=int, default=21,
                   help="Random seed for trainset shuffle (matches simba.py / bootstrapfewshot.py / mipro.py)")
    # ---- starting content ----
    p.add_argument("--instruction-file", default="instructions.md",
                   help="File whose contents become the starting signature instruction "
                        "(empty string to use the baseline NL2PLNSignature instruction)")
    p.add_argument("--pln-spec-file", default="chainer_analysis.txt",
                   help="File whose contents become the pln_spec input value "
                        "(empty string to use PeTTaChainer's default LLM_RULE_SPEC.md)")
    p.add_argument("--input", default=None,
                   help="Optional checkpoint to bootstrap from (typically a SIMBA "
                        "checkpoint); ignored silently if the file does not exist.")
    p.add_argument("--output", default="programs/gepa.json",
                   help="Where to save the optimized program")
    # ---- logging ----
    p.add_argument("--log-dir", default="gepa_logs",
                   help="Directory for GEPA run logs (GEPA's own structured logs)")
    p.add_argument("--log-level", default="info",
                   choices=["debug", "info", "warning", "error"],
                   help="Logging verbosity (default: info)")
    p.add_argument("--log-file", default="/tmp/gepa.log",
                   help="Path to log file; mirrors stdout/stderr for Colab "
                        "where streaming cell output gets truncated.")
    return p.parse_args()


def main():
    args = parse_args()

    setup_logging(args.log_file, args.log_level)
    print(f"  Logging to {args.log_file} (level={args.log_level})")

    lm_kwargs = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else {}
    task_lm = dspy.LM(args.model, **lm_kwargs)
    reflection_lm = dspy.LM(args.reflection_model, **lm_kwargs) if args.reflection_model else task_lm
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

    # Shuffle so GEPA's minibatch sampling sees phenomenon-diverse samples
    # rather than clustering on the first few phenomena from the dataset.
    import random
    random.Random(args.seed).shuffle(trainset)

    if args.valset:
        valset = build_examples_from_file(args.valset)
        print(f"  Loaded {len(valset)} validation examples from {args.valset}")
    else:
        # Preserve the original behavior of using the trainset as the valset.
        valset = trainset

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
            # Manual instruction override wipes the auto-injected pln_spec
            # section.  Re-inject it so the LM still sees the spec.
            module._inject_pln_spec()
            print(f"  Overrode instruction from {instr_path} ({len(new_instruction):,} chars)")
        else:
            logger.warning("instruction file %s not found; using default", instr_path)

    teleprompter = GEPA(
        metric=difficulty_metric,
        reflection_lm=reflection_lm,
        num_threads=args.num_threads,
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=args.reflection_minibatch_size,
        track_stats=True,
        track_best_outputs=True,
        log_dir=args.log_dir,
        gepa_kwargs={"stop_callbacks": [
            MaxMetricCallsStopper(args.max_metric_calls),
            ScoreThresholdStopper(args.score_threshold),
        ]},
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Wrap compile() so that an uncaught exception still leaves a saved
    # snapshot.  GEPA is the most expensive optimizer in this repo, so
    # losing the entire run to a mid-flight crash (Colab recycle, quota
    # exhaustion that DOES propagate, etc.) is especially costly.
    compile_succeeded = False
    try:
        module = teleprompter.compile(
            module,
            trainset=trainset,
            valset=valset,
        )
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

#for i in range(0,1):
#
#    teleprompter = GEPA(metric=difficulty_metric
#                   ,reflection_lm=dspy.LM(model,temperature=1.0)
#                   ,num_threads=10
#                   ,max_metric_calls=total_metric_calls
#                   ,reflection_minibatch_size=3
#                   ,track_stats=True
#                   ,track_best_outputs=True
#                   ,log_dir=f"gepa_logs{i}"
#                   ,gepa_kwargs={"stop_callbacks": [MaxMetricCallsStopper(total_metric_calls),ScoreThresholdStopper(0.9)]}
#                   )
#
#    if i > 1:
#        module.load(f"programs/auto{i - 1}_cnting.json")
#
#    #trainset = [dataset[i]]
#    trainset = dataset[:(i + 1)]
#    valset = dataset[:(i + 1)]
#    module = teleprompter.compile(
#        module,
#        trainset=trainset,
#        valset=valset,
#    )
#
#    print(pformat(module.detailed_results, width=300, indent=2))
#    with open(f"programs/auto{i}dr_cnting.json", "w") as f:
#        f.write(str(module.detailed_results))
#    module.save(f"programs/auto{i}_cnting.json")
#
#    passed = False
#    for val_scores in module.detailed_results.val_subscores:
#        all = True
#        for score in val_scores:
#            all = all and (score > 0.6)
#        passed = all or passed
#    if not passed:
#        print("Module not good enough for all samples. Stopping")
#        break
