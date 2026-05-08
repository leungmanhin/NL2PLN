import argparse
import os
import dspy
import json
import mlflow
import logging
from pprint import pformat
from dspy.teleprompt import SIMBA
from dspy.utils.callback import BaseCallback
from pathlib import Path

from nl2pln import NL2PLNModule , difficulty_metric , build_examples_from_file
from logging_utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run SIMBA optimization on NL2PLNModule"
    )
    p.add_argument("--model", default="openai/gpt-5.4-mini",
                   help="LiteLLM model id used for the task LM (translator + judge).  "
                        "Also used as SIMBA's prompt-candidate LM by default unless "
                        "--reflection-model is set.")
    p.add_argument("--reflection-model", default=None,
                   help="LiteLLM model id for SIMBA's prompt-candidate LM (used by "
                        "append_a_rule to propose new rules from failure traces).  "
                        "Recommended: a stronger model than --model — rule-proposer "
                        "is called few times but benefits from sharper synthesis.  "
                        "If unset, uses --model.")
    p.add_argument("--reasoning-effort", default="high",
                   help="Reasoning effort for reasoning-capable models: none, low, medium, high, xhigh.  "
                        "Applies to both task LM and reflection LM.")
    p.add_argument("--max-tokens", type=int, default=32768,
                   help="Output token cap for both task LM and reflection LM.  "
                        "DSPy/LiteLLM default is ~4k which routinely truncates "
                        "reasoning models at high effort (reasoning tokens count "
                        "against this budget).  Symptom of too-low: empty "
                        "pred.statements/pred.queries → puzzles land in hard_zero.")
    p.add_argument("--dataset", default="data/generated.json",
                   help="Training dataset path (default: bootstrap-generated training data)")
    p.add_argument("--num-threads", type=int, default=10,
                   help="Parallel LM calls during optimization")
    p.add_argument("--bsize", type=int, default=8,
                   help="SIMBA mini-batch size")
    p.add_argument("--max-steps", type=int, default=8,
                   help="Number of mini-batch iterations (training budget)")
    p.add_argument("--num-candidates", type=int, default=6,
                   help="Trajectories sampled per example per batch (also caps candidates built per batch)")
    p.add_argument("--max-demos", type=int, default=4,
                   help="Maximum few-shot demos per predictor; oldest are probabilistically dropped beyond this")
    p.add_argument("--seed", type=int, default=21,
                   help="Random seed for trainset shuffle (matches bootstrapfewshot.py / mipro.py / gepa_opt.py)")
    p.add_argument("--instruction-file", default="",
                   help="File whose contents become the signature instruction "
                        "(empty string to use the baseline NL2PLNSignature instruction)")
    p.add_argument("--pln-spec-file", default="bootstrap/chainer_analysis.txt",
                   help="File whose contents become the pln_spec input value "
                        "(empty string leaves nl2pln.pln_spec as currently set, default empty)")
    p.add_argument("--input", default=None,
                   help="Optional checkpoint to resume from; ignored silently if the file does not exist")
    p.add_argument("--output", default="programs/simba.json",
                   help="Where to save the optimized program")
    p.add_argument("--log-level", default="info",
                   choices=["debug", "info", "warning", "error"],
                   help="Logging verbosity (default: info)")
    p.add_argument("--log-file", default="/tmp/simba.log",
                   help="Path to log file; mirrors stdout/stderr for Colab "
                        "where streaming cell output gets truncated.")
    return p.parse_args()


def main():
    args = parse_args()

    setup_logging(args.log_file, args.log_level)
    print(f"  Logging to {args.log_file} (level={args.log_level})")

    lm_kwargs = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else {}
    lm_kwargs["max_tokens"] = args.max_tokens
    dspy.configure(lm=dspy.LM(args.model, **lm_kwargs))
    reflection_model_id = args.reflection_model or args.model
    if args.reflection_model:
        print(f"  Reflection LM: {args.reflection_model} (task LM: {args.model})")

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

    dataset = build_examples_from_file(args.dataset)
    print(f"  Loaded {len(dataset)} training examples from {args.dataset}")

    # Shuffle so SIMBA's mini-batch iteration sees phenomenon-diverse samples
    # rather than clustering on the first few phenomena from the dataset.
    import random
    random.Random(args.seed).shuffle(dataset)

    module = NL2PLNModule()
    if args.input:
        input_path = Path(args.input)
        if input_path.exists():
            module.load(str(input_path))
            print(f"  Loaded checkpoint from {input_path}")
        else:
            logger.info("No checkpoint found at %s; training from uninitialized module.", input_path)

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

    teleprompter = SIMBA(
        metric=difficulty_metric,
        prompt_model=dspy.LM(reflection_model_id, temperature=1.0, **lm_kwargs),
        bsize=args.bsize,
        max_steps=args.max_steps,
        num_candidates=args.num_candidates,
        max_demos=args.max_demos,
        num_threads=args.num_threads,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    module = teleprompter.compile(module, trainset=dataset)
    module.save(str(output_path))
    print(f"  Saved optimized program to {output_path}")


if __name__ == "__main__":
    main()
