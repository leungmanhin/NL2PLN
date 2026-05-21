import dspy
from nl2pln import NL2PLNModule, difficulty_metric, build_examples_from_file

dspy.configure(lm=dspy.LM("openai/gpt-5.4-mini", reasoning_effort="high"))

dataset = build_examples_from_file("data/all.json")

def evaluate(module, label):
    total = 0.0
    for ex in dataset:
        pred = module(**ex.inputs())
        score = difficulty_metric(ex, pred).score
        total += score
        print(f"{ex.inputs()}\n{pred}\nScore: {score}\n")
    avg = total / len(dataset)
    print(f"{label}: {avg:.4f}")
    return avg

# The SIMBA artifact
optimized = NL2PLNModule()
optimized.load("programs/simba_all_roman.json")
evaluate(optimized, "simba-roman")

# Custom prompt: use the bootstrap artifacts instead of the baseline spec.
#   signature instruction  <- conversion_guidelines.txt
#   pln_spec input field   <- chainer_analysis.txt
# NL2PLNModule.forward() reads pln_spec from the module-level constant in
# nl2pln.py, so we monkey-patch it. Roman's program above already ran under
# the default (PeTTaChainer's LLM_RULE_SPEC.md).
with open("conversion_guidelines.txt", encoding="utf-8") as f:
    conversion_guidelines = f.read()
with open("chainer_analysis.txt", encoding="utf-8") as f:
    chainer_analysis = f.read()

import nl2pln
nl2pln.pln_spec = chainer_analysis

custom = NL2PLNModule()
# DSPy 3.2's ChainOfThought doesn't expose .signature on itself; iterate
# via named_predictors() to reach the underlying Predict objects that do.
for _, predictor in custom.named_predictors():
    predictor.signature = predictor.signature.with_instructions(conversion_guidelines)
evaluate(custom, "bootstrap-prompt")
