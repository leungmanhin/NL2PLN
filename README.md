## Prerequisites

- Python ≥ 3.10
- SWI-Prolog ≥ 9.3 (required by PeTTa)

## Install

Clone the three repos into the same parent directory, then sync dependencies from NL2PLN:

```bash
git clone https://github.com/patham9/PeTTa.git
git clone https://github.com/rTreutlein/PeTTaChainer.git
git clone https://github.com/rTreutlein/NL2PLN.git
cd NL2PLN
uv sync
```

`uv sync` installs PeTTa and PeTTaChainer editable from the sibling checkouts, along with the remaining Python dependencies.

Have a look at `src/usage_example.py` for how to use the model.

- `nl2pln.py` is the main module that contains the NL2PLN model for training.
- `simba.py` contains training using the Simba optimizer.
- `gepa.py` contains training using the GEPA optimizer.
