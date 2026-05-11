# NL2PLN

An optimization pipeline that produces a semantic parser for converting
natural-language sentences into PLN expressions (statements and queries)
that work out-of-the-box with a given chainer (logical reasoner).  The
pipeline takes a chainer as input and produces an NL→PLN DSPy program
optimized against that chainer, a specific LM, and a training-data
distribution.

## Prerequisites

- Python ≥ 3.10
- SWI-Prolog ≥ 9.3 (required by PeTTa)

## Install

Clone this repo and its siblings into the same parent directory, then
sync dependencies:

```bash
git clone https://github.com/patham9/PeTTa.git
git clone https://github.com/rTreutlein/PeTTaChainer.git
git clone https://github.com/rTreutlein/NL2PLN.git
cd NL2PLN
uv sync
```

`uv sync` installs PeTTa and PeTTaChainer editable from the sibling
checkouts, along with the remaining Python dependencies.

## Pipeline

The full optimization workflow:

1. **Bootstrap** — analyze the chainer's source code and enumerate the
   linguistic phenomena the parser must handle.  Produces
   `bootstrap/chainer_analysis.txt` (the `pln_spec` downstream stages
   consume) and `bootstrap/linguistic_phenomena.txt`.

   ```bash
   python src/bootstrap_chainer.py --chainer ../PeTTaChainer
   ```

2. **Generate training data** — synthesize per-phenomenon training
   examples (NL sentences + question/expected-answer pairs) using a
   generator + verifier LLM.  Produces `data/generated.json`.

   ```bash
   python src/generate_data.py
   ```

3. **Optimize** — train an NL→PLN program.  Four DSPy teleprompters
   are available; pick one (`--help` on each shows the full flag
   surface):

   ```bash
   python src/simba.py             # SIMBA — append rules to instruction
   python src/gepa_opt.py          # GEPA — reflective instruction rewriting
   python src/bootstrapfewshot.py  # BootstrapFewShot — demos only
   python src/mipro.py             # MIPROv2 — instructions + demos jointly
   ```

4. **Evaluate** — score one or more programs on a dataset, with
   per-program `pln_spec` selection:

   ```bash
   python scripts/build_eval_set.py        # (optional) build data/eval.json
   python scripts/eval_program.py programs/simba.json --dataset data/eval.json
   ```

## Deploying an optimized program

See `scripts/usage_example.py` for an end-to-end demo: load a saved
program, run single-puzzle inference, and feed the resulting PLN to
PeTTaChainer.  A bundled program (`programs/simba_all.json`) is included
as a ready-made example.
