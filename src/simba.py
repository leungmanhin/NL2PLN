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

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run SIMBA optimization on NL2PLNModule"
    )
    p.add_argument("--model", default="openai/gpt-5.4-mini",
                   help="LiteLLM model id used for both task LM and SIMBA's prompt-candidate LM")
    p.add_argument("--reasoning-effort", default="high",
                   help="Reasoning effort for reasoning-capable models: none, low, medium, high, xhigh")
    p.add_argument("--dataset", default="data/all.json",
                   help="Path to dataset JSON")
    p.add_argument("--num-threads", type=int, default=10,
                   help="Parallel LM calls during optimization")
    p.add_argument("--bsize", type=int, default=8,
                   help="SIMBA batch size")
    p.add_argument("--input", default="programs/simba_all2_gepa.json",
                   help="Optional checkpoint to resume from; ignored silently if the file does not exist")
    p.add_argument("--output", default="programs/simba_all.json",
                   help="Where to save the optimized program")
    return p.parse_args()


def main():
    args = parse_args()

    lm_kwargs = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else {}
    dspy.configure(lm=dspy.LM(args.model, **lm_kwargs))

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(uri=tracking_uri)
        mlflow.set_experiment("DSPy-Optimization")
        mlflow.dspy.autolog(
            log_compiles=True,    # Track optimization process
            log_evals=True,       # Track evaluation results
            log_traces_from_compile=True  # Track program traces during optimization
        )

    dataset = build_examples_from_file(args.dataset)

    module = NL2PLNModule()
    checkpoint_path = Path(args.input)
    if checkpoint_path.exists():
        module.load(str(checkpoint_path))
    else:
        logger.info("No checkpoint found at %s; training from uninitialized module.", checkpoint_path)

    teleprompter = SIMBA(
        metric=difficulty_metric,
        prompt_model=dspy.LM(args.model, temperature=1.0, **lm_kwargs),
        bsize=args.bsize,
        num_threads=args.num_threads,
    )

    module = teleprompter.compile(module, trainset=dataset)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    module.save(str(output_path))


if __name__ == "__main__":
    main()

#for i in range(0,2):
#
#    teleprompter = SIMBA(
#        metric=difficulty_metric,
#        prompt_model=dspy.LM(optmodel, temperature=1.0),
#        bsize=i+1,
#        num_threads=10,
#    )
#
#    if i > 1:
#        module.load(f"programs/sauto{i - 1}_andres.json")
#
#    #trainset = [dataset[i]]
#    trainset = dataset[:(i + 1)]
#    module = teleprompter.compile(
#        module,
#        trainset=trainset,
#    )
#
#    module.save(f"programs/sauto{i}_andres.json")
