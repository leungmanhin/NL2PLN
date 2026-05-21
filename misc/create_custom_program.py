import dspy
from nl2pln import NL2PLNModule

# Option 1: Start from the optimized artifact, override just the instructions
m = NL2PLNModule()
m.load("programs/simba_all.json")
m.nl2pln.signature = m.nl2pln.signature.with_instructions("YOUR PROMPT TEXT HERE")
# Keeps SIMBA's demos in place; only the instruction changes

# Option 2: Fresh module with your instructions, no demos
m2 = NL2PLNModule()
m2.nl2pln.signature = m2.nl2pln.signature.with_instructions("YOUR PROMPT TEXT HERE")
# Baseline-style: your instruction vs. the un-optimized default instruction
