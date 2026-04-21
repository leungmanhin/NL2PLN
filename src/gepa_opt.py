import argparse
import os
import dspy
import json
from gepa.utils.stop_condition import MaxMetricCallsStopper, ScoreThresholdStopper
import mlflow
from pprint import pformat
from dspy.teleprompt import GEPA
from dspy.utils.callback import BaseCallback
from pathlib import Path

from nl2pln import NL2PLNModule , difficulty_metric , build_examples_from_file

import logging
logger = logging.getLogger(dspy.teleprompt.gepa.gepa.__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run GEPA optimization on NL2PLNModule (bootstraps from a SIMBA checkpoint)"
    )
    p.add_argument("--model", default="openai/gpt-5.4-mini",
                   help="LiteLLM model id used for both task LM and GEPA's reflection LM")
    p.add_argument("--reasoning-effort", default="high",
                   help="Reasoning effort for reasoning-capable models: none, low, medium, high, xhigh")
    p.add_argument("--dataset", default="data/all.json",
                   help="Path to dataset JSON")
    p.add_argument("--num-threads", type=int, default=10,
                   help="Parallel LM calls during optimization")
    p.add_argument("--max-metric-calls", type=int, default=1000,
                   help="GEPA evaluation budget (total metric calls)")
    p.add_argument("--reflection-minibatch-size", type=int, default=16,
                   help="Minibatch size for GEPA reflective mutation")
    p.add_argument("--score-threshold", type=float, default=0.9,
                   help="Early-stop threshold: stop when a candidate scores >= this")
    p.add_argument("--log-dir", default="gepa_logs_all",
                   help="Directory for GEPA run logs")
    p.add_argument("--input", default="programs/simba_all.json",
                   help="Checkpoint to bootstrap from (typically SIMBA's output)")
    p.add_argument("--output", default="programs/simba_all2_gepa.json",
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

    #shutil.rmtree('gepa_logs')

    module = NL2PLNModule()
    module.load(args.input)

    teleprompter = GEPA(
        metric=difficulty_metric,
        reflection_lm=dspy.LM(args.model, **lm_kwargs),
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
    module = teleprompter.compile(
        module,
        trainset=dataset,
        valset=dataset,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    module.save(str(output_path))


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
