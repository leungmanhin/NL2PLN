# NL to logic conversion instructions

You are converting English natural language (sentences or questions) into logical expressions (statements or queries) for a chainer/reasoner.

## Chainer syntax reference

# PeTTaChainer source analysis

## 1) Expression format

The Python API expects **bare MeTTa expressions** as input strings.

### Public bare format
The core outer envelope used by both statements and queries is:

```metta
(: <proof-id> <proposition-or-pattern> <truth-value-or-pattern>)
```

Examples of bare inputs passed to the API:

```metta
(: f1 (Dog fido) (STV 1.0 1.0))
(: $prf (Dog fido) $tv)
(: rule1
   (Implication
      (Premises (Dog $x) (Not (Cat $x)))
      (Conclusions (Mammal $x)))
   (STV 1.0 1.0))
```

### Internal runtime wrappers
Do **not** pass these as external API input; the chainer inserts them internally:

```metta
!(eval <bare-expression>)
!(compileadd <kb> <evaluated-statement>)
!(query <steps> <kb> <evaluated-query>)
```

So the rule for downstream consumers is:

- **caller input**: bare MeTTa expression
- **internal execution**: wrapped `!(eval ...)`, then `!(compileadd ...)` or `!(query ...)`

The public methods `add_atom()` and `query()` both evaluate the input first, then hand the evaluated bare expression to the internal MeTTa runtime.

## 2) Built-in operators / combinators

The public docs and source show these main surface forms.

### Statement / proof-atom envelope
```metta
(: <proof-id> <proposition> <tv>)
```

This is the storage/query envelope for facts and rules.

### `Implication`
```metta
(Implication (Premises ...) (Conclusions ...))
```
Represents an if-then rule.

### `Premises`
```metta
(Premises premise1 premise2 ...)
```
Rule antecedent list.

### `Conclusions`
```metta
(Conclusions conclusion1 conclusion2 ...)
```
Rule consequent list.

### `Compute`
```metta
(Compute f (arg1 arg2 ...) -> $out)
```
Compute a function result and bind it.

### `FoldAll` / `FoldAllValue`
```metta
(FoldAll pattern value init fold-fn -> out)
(FoldAllValue pattern init fold-fn -> out)
```
Aggregate over matching facts.

### `Not`
```metta
(Not expr)
```
Negation / negative premise.

### `GreaterThan` and alias `>`
```metta
(GreaterThan $distA 5)
(GreaterThan $distA $distB)
```
Docs say `>` is an alias/sugar for the same comparison family.

### `MapDist`, `Map2Dist`, `AverageDist`
```metta
(MapDist f (DistFactA ... $inDist) $inDist -> $outDist)
(Map2Dist f (DistFactA ... $distA) $distA (DistFactB ... $distB) $distB -> $outDist)
(AverageDist (DistFactPattern ... $inDist) $inDist -> $outDist)
```

These are helper premises for distribution-valued facts.

### Lower-level / compiled distribution forms seen in source
These are not the main user-facing rule skeleton, but they are part of the chainer’s supported operator set and should be treated as structural, not domain predicates:

- `DistGreaterThanFormula`
- `DistGreaterThanDistFormula`
- `ParticleMap`
- `ParticleMap2`
- `ParticleAddBernoulliFromSTV`

### Utility / store operators
Also documented in the spec:

- `ParticleStoreCount`
- `ParticleStoreClear`
- `ParticleStorePruneKB`

## 3) Truth values

The chainer distinguishes **truth uncertainty** from **value uncertainty**.

### Truth uncertainty
```metta
(STV strength confidence)
```

- `strength` is the truth / belief mass
- `confidence` is evidence / reliability
- both are intended to be in `[0, 1]`

`STV` is for the truth of a proposition, not for numeric measurement uncertainty.

### Value uncertainty / distributions
Supported distribution-like truth/value forms:

```metta
(NatDist ((value probability) ...))
(FloatDist ((value probability) ...))
(ParticleDist <ref>)
(ParticleDist <ref> <scale>)
(PointMass x)
(ParticleFromNormal mu sigma)
(ParticleFromPairs ((x1 w1) (x2 w2) ...))
```

Notes:

- `NatDist` and `FloatDist` are exact discrete distributions.
- `ParticleDist` is an opaque reference backed by the particle store.
- `PointMass` is a degenerate distribution.
- `ParticleFromNormal` and `ParticleFromPairs` are particle-based constructors.
- The validator explicitly accepts the above shapes.

## 4) Rule templates

Rules are just statements whose proposition is an `Implication`.

Exact template:

```metta
(: <rule-name>
   (Implication
      (Premises
         premise1
         premise2
         ...)
      (Conclusions
         conclusion1
         conclusion2
         ...))
   (STV <strength> <confidence>))
```

The source/docs use the same outer `(: ... )` envelope for facts and rules.

## 5) Query patterns

Query patterns use the same outer envelope, but the proof-id slot must be a **variable**.

Typical query shape:

```metta
(: $prf <pattern> $tv)
```

Examples:

```metta
(: $prf (Dog fido) $tv)
(: $prf (AreaDist rectA $areaDist) $tv)
```

Important source-derived constraints:

- `check_query()` requires the proof-id slot to be a variable.
- The validator does **not** impose the same deep truth-value restrictions on queries that it does on statements.
- In intended usage, the truth-value slot is usually also a variable (`$tv`) so the query can return it.

## 6) Naming conventions

Observed conventions in docs/examples:

- **Variables**: prefixed with `$`
  - examples: `$prf`, `$tv`, `$x`, `$dist`, `$avgDist`
- **Predicates / relations**: usually CamelCase
  - examples: `Dog`, `HeightDist`, `AvgHeightDist`, `CountryHeightDist`, `Rectangle`, `Group`
- **Constants / entities**: plain atoms, often lowercase or mixed case
  - examples: `fido`, `alice`, `bob`, `carol`, `room1`, `g1`, `rectA`, `countryA`
- **Rule / proof ids**: opaque labels, often camelCase or descriptive identifiers
  - examples: `f1`, `r1`, `avgHeightDistG1Rule`

There is no quoted-string entity syntax in the source examples.

## 7) Constraints & pitfalls

### Do not use internal runtime wrappers as external input
External callers should pass bare expressions, not:

```metta
!(eval ...)
!(compileadd ...)
!(query ...)
```

### Statement validation is shallow but strict on the envelope
For statements, the validator checks that the evaluated expression has the outer `(: ...)` shape, that the proof-id is not a variable, and that the TV matches one of the supported constructors.

### Query validation
For queries, the validator checks that the proof-id is a variable.

### No dedicated `ForAll` / `Exists` user syntax
The repo contains unrelated internal/Prolog `forall` / `exists` occurrences, but there is no public bare-expression quantifier operator in the chainer API.

### Avoid confusing internal proof-tree tokens with domain predicates
Source examples contain internal tokens such as:

- `CPU` / `cpu`
- `conjunction`
- `rule-proof`
- `fact-ev`
- `LikelierThan`

These are proof/runtime artifacts, not user-defined domain relations.

### Beware of `>` vs `GreaterThan`
`GreaterThan` is the canonical documented head; `>` is documented as an alias/sugar.

### Bare syntax is prefix s-expression style
No commas or infix statement syntax is used in the public API.

## 8) Predicate extraction

For bare-expression predicate/relation extraction, use a regex that captures identifier-like heads after `(` and then filters out structural symbols.

Recommended regex:

```python
r'\(\s*([A-Za-z][A-Za-z0-9_+-]*)'
```

Why this works:

- it captures heads from statements, queries, and nested subexpressions
- it works with Python `re` on single-line expression strings
- it captures normal relation names like `Dog`, `HeightDist`, `Map2Dist`
- built-ins are then removed using an exclusion list

## 9) Quantification and scope

There is **no dedicated quantifier syntax** like `ForAll` / `Exists` in the public bare language.

Quantifier-like meaning is expressed indirectly:

- **existential** meaning: via query variables and proof search
- **universal / rule-like** meaning: via variables in rule premises and implications
- **count/cardinality** style constraints: via aggregation helpers such as `FoldAll` plus comparison operators

Scope is syntactic/nested:

- variables live in the nested MeTTa expression structure
- `Not` scopes over its nested expression
- premise helpers such as `Compute`, `FoldAll`, `MapDist`, and `AverageDist` scope their binders using the surrounding prefix form and `->` output binder

## 10) Entity identity and representation

There is no built-in entity identity system beyond symbol identity.

Implications:

- the same atom spelling refers to the same symbol/entity
- different spellings are different entities
- the source does not define a canonicalization or disambiguation layer for names

Recommended practice:

- use stable, unique constants for named entities
- if surface names are ambiguous, keep a separate ID and optionally a `Name`/`Alias` relation in your own ontology
- do not rely on natural-language surface forms alone if distinct real-world entities may share the same name

Observed idioms for instance/class membership are just unary predicates:

```metta
(Dog fido)
(Person alice)
(Rectangle rectA)
(Group g1)
```

No special first/second-person pronoun convention is defined in the source. If the referent is unknown, skip the pronoun or ground it externally.

---

## Conversion guidelines

PeTTaChainer English-to-logic conversion guidelines

1. Core surface forms

Use only bare MeTTa expressions as external input. Do not add internal wrappers such as `!(eval ...)`, `!(compileadd ...)`, or `!(query ...)`.

There are three public patterns:

Fact / assertion:
```metta
(: factId <proposition> (STV s c))
```

Rule:
```metta
(: ruleId
   (Implication
      (Premises premise1 premise2 ...)
      (Conclusions conclusion1 conclusion2 ...))
   (STV s c))
```

Query:
```metta
(: $prf <pattern> $tv)
```

Rules:
- In statements, the proof id must be a constant atom, not a variable.
- In queries, the proof id must be a variable, typically `$prf`.
- Prefer `GreaterThan` over the `>` alias for consistency.
- The truth-value slot in queries should usually be a variable, typically `$tv`.

2. Naming conventions

2.1 Predicates
- Use CamelCase for predicate names.
- Lemmatize English words to base form:
  - “dogs” -> `Dog`
  - “barked” -> `Bark`
  - “cities” -> `City`
- Omit articles, auxiliaries, and tense morphology from predicate names.
- Keep predicate names semantically atomic: usually 1 to 3 English words.
- Good predicate shapes:
  - `Dog`
  - `City`
  - `Own`
  - `PartOf`
  - `Before`
  - `Recipient`
  - `Height`
  - `HeightDist`

2.2 Constants / entities
- Use stable lowercase atoms, optionally with underscores:
  - `fido`
  - `paris_france`
  - `book1`
  - `e1`
- For multiword names, normalize to one atom:
  - “New York City” -> `new_york_city`
  - “Big Apple” -> `big_apple`
- Do not use quoted strings; the analyzed chainer does not define quoted-string entity syntax.

2.3 Variables
- Prefix all variables with `$`.
- Use descriptive variable names when possible:
  - `$x`, `$y` for general entities
  - `$e` for events
  - `$t` for times
  - `$n` for numbers/counts
  - `$d` for distributions
  - `$prf` for query proof id
  - `$tv` for query truth value

2.4 Rule / proof ids
- Use opaque but descriptive constants:
  - `f_paris_city`
  - `r_whale_mammal`
  - `r_cut_summary`

3. Entity representation and identity

3.1 Identity semantics
There is no built-in identity beyond symbol identity.
- The same atom spelling means the same entity.
- Different spellings mean different entities.
- Therefore: resolve identity before emission whenever possible.

3.2 Named entities
Use a stable unique entity id, not the surface name alone when ambiguity is possible.

Example:
```metta
(: f1 (Person alex_smith_1) (STV 1.0 1.0))
(: f2 (Name alex_smith_1 alex) (STV 1.0 1.0))
(: f3 (Person alex_smith_2) (STV 1.0 1.0))
(: f4 (Name alex_smith_2 alex) (STV 1.0 1.0))
```

3.3 Appellations and aliases
Use separate relations such as `Name` or `Alias` if the ontology needs them. Do not rely on surface form equality for disambiguation.

3.4 First- and second-person pronouns
The source defines no special semantics for `I`, `we`, or `you`.
- If dialogue context supplies grounded participants, map them to stable constants such as `speaker_1`, `addressee_1`, `speaker_group_1`.
- If the referent is unknown, do not invent a permanent world entity from the pronoun alone; skip assertion or leave resolution to an upstream context module.

4. Predicate atomicity: mandatory discipline

Every predicate name must denote one semantic primitive, not a whole mini-sentence.

Why this matters:
PeTTaChainer composes proofs by matching predicate structures. If a predicate name hides comparison, counting, argument structure, or entity identity inside the name, the reasoner cannot reuse that internal structure compositionally.

4.1 General rule
Bad predicates encode:
- numbers in the name
- thresholds in the name
- multiple concepts fused into one name
- named entities fused into the predicate
- whole answers instead of reusable relations

Good predicates expose:
- entities as arguments
- numbers as arguments
- comparisons via `GreaterThan`
- negation via `Not`
- conjunction via multiple premises or multiple facts
- conditionality via `Implication`

4.2 BAD -> GOOD examples

A. Numbers embedded in the predicate name

BAD:
```metta
(: f1 (Height180Alice) (STV 1.0 1.0))
```

GOOD:
```metta
(: f1 (Height alice 180 centimeter) (STV 1.0 1.0))
```

B. Threshold embedded in the name

BAD:
```metta
(: f1 (HeavyOver5Kg bag1) (STV 1.0 1.0))
```

GOOD:
```metta
(: f1 (Weight bag1 7 kilogram) (STV 1.0 1.0))
(: r1
   (Implication
      (Premises
         (Weight $x $w kilogram)
         (GreaterThan $w 5))
      (Conclusions
         (Heavy $x)))
   (STV 1.0 1.0))
```

C. Multi-concept concatenation in one predicate

BAD:
```metta
(: f1 (GiveBookToMary john) (STV 1.0 1.0))
```

GOOD:
```metta
(: f1 (GiveEvent e1) (STV 1.0 1.0))
(: f2 (Agent e1 john) (STV 1.0 1.0))
(: f3 (Recipient e1 mary) (STV 1.0 1.0))
(: f4 (Theme e1 book1) (STV 1.0 1.0))
```

D. Entity-specific comparison fused into the name

BAD:
```metta
(: f1 (TomTallerThanSam) (STV 1.0 1.0))
```

GOOD:
```metta
(: f1 (Height tom 190 centimeter) (STV 1.0 1.0))
(: f2 (Height sam 175 centimeter) (STV 1.0 1.0))
(: r1
   (Implication
      (Premises
         (Height $x $hx centimeter)
         (Height $y $hy centimeter)
         (GreaterThan $hx $hy))
      (Conclusions
         (TallerThan $x $y)))
   (STV 1.0 1.0))
```

E. Whole-event content baked into one predicate

BAD:
```metta
(: f1 (ReadBookYesterdayQuietly alice book1) (STV 1.0 1.0))
```

GOOD:
```metta
(: f1 (ReadEvent e1) (STV 1.0 1.0))
(: f2 (Agent e1 alice) (STV 1.0 1.0))
(: f3 (Theme e1 book1) (STV 1.0 1.0))
(: f4 (Time e1 yesterday) (STV 1.0 1.0))
(: f5 (Manner e1 quietly) (STV 1.0 1.0))
```

Self-check:
If the predicate name sounds like the whole English clause, split it.

5. Decomposition strategies

5.1 Simple facts
If the sentence is a simple classification or property with no extra structure, use one atomic fact:
```metta
(: f1 (City paris_france) (STV 1.0 1.0))
(: f2 (Red car1) (STV 1.0 1.0))
```

5.2 Complex clauses
If a clause contains roles, time, location, manner, modality, cause, purpose, or embedding, introduce an event/state/proposition constant and decompose into linked facts.

Recommended general role predicates:
- `Agent`
- `Patient`
- `Theme`
- `Recipient`
- `Beneficiary`
- `Instrument`
- `Experiencer`
- `Source`
- `Goal`
- `Path`
- `Location`
- `Time`
- `Manner`
- `Frequency`
- `Tense`
- `Aspect`
- `Cause`
- `Purpose`
- `Content`

Example:
“Alice cut the bread with a knife in the kitchen yesterday.”
```metta
(: f1 (CutEvent e1) (STV 1.0 1.0))
(: f2 (Agent e1 alice) (STV 1.0 1.0))
(: f3 (Patient e1 bread1) (STV 1.0 1.0))
(: f4 (Instrument e1 knife1) (STV 1.0 1.0))
(: f5 (Location e1 kitchen1) (STV 1.0 1.0))
(: f6 (Time e1 yesterday) (STV 1.0 1.0))
```

5.3 Derived summary relations
Because queries are typically single-pattern, add rules that summarize decomposed structures into query-friendly relations.

Example:
```metta
(: r_cut_summary
   (Implication
      (Premises
         (CutEvent $e)
         (Agent $e $a)
         (Patient $e $p))
      (Conclusions
         (Cut $a $p)))
   (STV 1.0 1.0))
```

Then query:
```metta
(: $prf (Cut alice $x) $tv)
```

6. Quantification and scope

6.1 Universal quantification
There is no `ForAll`. Express universal/generic rule-like meaning with variables in an `Implication`.

“All whales are mammals.”
```metta
(: r1
   (Implication
      (Premises (Whale $x))
      (Conclusions (Mammal $x)))
   (STV 1.0 1.0))
```

6.2 Existential quantification in assertions
There is no `Exists`. Use a witness constant.

“A dog barked.”
```metta
(: f1 (Dog dog1) (STV 1.0 1.0))
(: f2 (Bark dog1) (STV 1.0 1.0))
```

If event structure is needed:
```metta
(: f1 (Dog dog1) (STV 1.0 1.0))
(: f2 (BarkEvent e1) (STV 1.0 1.0))
(: f3 (Agent e1 dog1) (STV 1.0 1.0))
```

6.3 Existential quantification in queries
Use variables in the query pattern.

“Did some dog bark?” / “Which dog barked?”
```metta
(: $prf (Bark $x) $tv)
```

If you must restrict to dogs and the query interface should remain single-pattern, derive a summary:
```metta
(: r1
   (Implication
      (Premises
         (Dog $x)
         (Bark $x))
      (Conclusions
         (BarkingDog $x)))
   (STV 1.0 1.0))
```

Then:
```metta
(: $prf (BarkingDog $x) $tv)
```

6.4 Cardinality
No dedicated cardinal quantifier exists. Use `FoldAll` / `FoldAllValue` when the deployment provides an appropriate fold/count function.

Template:
```metta
(: r_count
   (Implication
      (Premises
         (FoldAllValue (LaughingStudent $x) 0 countFn -> $n)
         (GreaterThan $n 2)
         (Not (GreaterThan $n 3)))
      (Conclusions
         (ExactlyN laughing_students 3)))
   (STV 1.0 1.0))
```

Guidelines:
- At least N: `GreaterThan $n N-1`
- At most N: `Not (GreaterThan $n N)`
- Exactly N over naturals: `GreaterThan $n N-1` and `Not (GreaterThan $n N)`

Important:
- `countFn` must already exist in the runtime environment; do not invent undocumented arithmetic operators.
- If no count fold function exists, do not fake the semantics by putting the number into the predicate name.

6.5 Scope with negation
Negation scopes over the nested expression:
```metta
(Not expr)
```

Use scope carefully:
- “A student did not leave” -> existential witness with negated predicate
```metta
(: f1 (Student student1) (STV 1.0 1.0))
(: f2 (Not (Leave student1)) (STV 1.0 1.0))
```

- “Every student did not leave” is usually not the same as “Not every student left”.
  - Universal negative reading: rule with negated conclusion
  - “Not every” reading: prefer a counterexample witness if available

For “Not every student left”, do not encode:
```metta
(: r_bad
   (Implication
      (Premises (Student $x))
      (Conclusions (Not (Leave $x))))
   (STV 1.0 1.0))
```
That means “No student left,” which is stronger.

Prefer, if a witness reading is licensed:
```metta
(: f1 (Student student1) (STV 1.0 1.0))
(: f2 (Not (Leave student1)) (STV 1.0 1.0))
```

If the exact scope cannot be represented safely, leave it for upstream handling rather than emitting a stronger or weaker false parse.

7. Built-in operators vs custom predicates

Use built-ins when the semantics matches a documented operator:
- `Implication` for conditionals and generics
- `Premises` / `Conclusions` for rule structure
- `Not` for negation
- `GreaterThan` for numeric comparison
- `Compute` for runtime-provided functions
- `FoldAll` / `FoldAllValue` for aggregation
- `MapDist`, `Map2Dist`, `AverageDist` for distribution-valued reasoning

Use custom predicates for domain concepts:
- `Dog`
- `Own`
- `CutEvent`
- `Agent`
- `Purpose`
- `BelongTo`

Do not replace built-ins with custom domain predicates:
- BAD: `IfDogThenMammal`
- GOOD: `Implication`

- BAD: `MoreThan5`
- GOOD: `GreaterThan`

- BAD: `NotCat`
- GOOD: `(Not (Cat ...))`

8. Truth value assignment

8.1 Default factual assertions
For plain asserted facts from the source sentence, use:
```metta
(STV 1.0 1.0)
```
unless the input explicitly indicates uncertainty.

8.2 Rules
Use `STV 1.0 1.0` for strict definitional or taxonomic rules.
Use lower confidence for defeasible generics or defaults when appropriate.

Example:
“Dogs bark” as a generic tendency may be less strict than taxonomy:
```metta
(: r1
   (Implication
      (Premises (Dog $x))
      (Conclusions (Bark $x)))
   (STV 0.9 0.7))
```

8.3 Negated assertions
Negated propositions still take an `STV`:
```metta
(: f1 (Not (Call alice)) (STV 1.0 1.0))
```

8.4 Value uncertainty vs truth uncertainty
Use `STV` for truth uncertainty.
Use distributions for uncertain values.

Preferred pattern:
- keep the proposition truthful with `STV`
- place the uncertain value inside the proposition

Example:
```metta
(: f1 (HeightDist alice (ParticleFromNormal 180 2)) (STV 1.0 1.0))
```

Do not confuse:
- truth uncertainty: “maybe true”
- value uncertainty: “true, but the measured value is distributed”

9. Query semantics and query granularity

9.1 Statements vs queries
Statements add knowledge.
Queries ask for proofs of a target pattern.

9.2 Rule-vs-query separation
Do not query a rule-shaped form when the English asks about a concrete derivable fact.

BAD for “Is moby a mammal?”:
```metta
(: $prf
   (Implication
      (Premises (Whale $x))
      (Conclusions (Mammal $x)))
   $tv)
```

GOOD:
Store the rule:
```metta
(: r1
   (Implication
      (Premises (Whale $x))
      (Conclusions (Mammal $x)))
   (STV 1.0 1.0))
```

Then query the target fact:
```metta
(: $prf (Mammal moby) $tv)
```

9.3 Query only the unknown
Do not emit a battery of queries for already asserted atoms.

BAD for “What did Alice cut?” after decomposing one cut event:
- query `CutEvent`
- query `Agent`
- query `Patient`
separately

GOOD:
Add a summary rule:
```metta
(: r1
   (Implication
      (Premises
         (CutEvent $e)
         (Agent $e $a)
         (Patient $e $p))
      (Conclusions
         (Cut $a $p)))
   (STV 1.0 1.0))
```

Then ask exactly:
```metta
(: $prf (Cut alice $x) $tv)
```

9.4 Query granularity asymmetry
Statements may be highly decomposed.
Queries should usually be a single target relation that the reasoner can derive from that decomposition.

If a natural-language question depends on a conjunction, prefer:
- decompose in the KB
- derive one queryable predicate by rule
- query that one predicate

10. Phenomenon-by-phenomenon mapping

1. Entity classification
- Use unary predicates for class membership.
- Use implication rules for class inclusion.

Examples:
```metta
(: f1 (City paris_france) (STV 1.0 1.0))
(: r1
   (Implication
      (Premises (Whale $x))
      (Conclusions (Mammal $x)))
   (STV 1.0 1.0))
```

2. Determiners and definiteness
- “a/an” in assertions -> introduce a new witness constant.
- “the” -> reuse an already established/resolved constant.
- “this/that” -> same entity handling as definites; optionally add `Proximal` / `Distal` if deixis matters.

Example:
```metta
(: f1 (Dog dog1) (STV 1.0 1.0))
(: f2 (Loud dog1) (STV 1.0 1.0))
(: f3 (Proximal dog1) (STV 1.0 1.0))
```

3. Properties and attributes
- Simple adjective/property -> unary predicate:
```metta
(: f1 (Fast car1) (STV 1.0 1.0))
(: f2 (Hot soup1) (STV 1.0 1.0))
```
- If the property has an explicit value or unit, use arguments:
```metta
(: f1 (Height alice 180 centimeter) (STV 1.0 1.0))
```

4. Actions and events with thematic roles
- For simple unmodified events, a direct predicate is acceptable:
```metta
(: f1 (Own sara bike1) (STV 1.0 1.0))
```
- For events with multiple roles/modifiers, use event decomposition.

Example:
```metta
(: f1 (CutEvent e1) (STV 1.0 1.0))
(: f2 (Agent e1 alice) (STV 1.0 1.0))
(: f3 (Patient e1 bread1) (STV 1.0 1.0))
(: f4 (Instrument e1 knife1) (STV 1.0 1.0))
```

5. Possession and ownership
- Use general binary predicates such as `Have`, `Own`, `BelongTo`.
- For possessive noun compounds that are part-whole, prefer `PartOf`.

Examples:
```metta
(: f1 (Have sara bike1) (STV 1.0 1.0))
(: f2 (BelongTo book1 omar) (STV 1.0 1.0))
(: f3 (PartOf engine1 car1) (STV 1.0 1.0))
```

6. Spatial relations
- Static relations: binary predicates such as `On`, `Under`, `Beside`, `In`, `Over`.
- Motion relations: event + `Source`, `Goal`, `Path`.

Examples:
```metta
(: f1 (Under key1 mat1) (STV 1.0 1.0))
(: f2 (Beside cafe1 station1) (STV 1.0 1.0))
(: f3 (FlyEvent e1) (STV 1.0 1.0))
(: f4 (Agent e1 plane1) (STV 1.0 1.0))
(: f5 (Over e1 city1) (STV 1.0 1.0))
```

7. Temporal relations and tense
- Use `Before`, `After`, `Time`, and `Tense`.
- Tense should usually be represented as a separate feature, not baked into the main predicate.

Examples:
```metta
(: f1 (ArriveEvent e1) (STV 1.0 1.0))
(: f2 (Time e1 noon) (STV 1.0 1.0))
(: f3 (Before e1 noon) (STV 1.0 1.0))
(: f4 (Tense e1 past) (STV 1.0 1.0))
```

8. Aspect and event viewpoint
- Use `Aspect` on an event.
- Suggested values: `progressive`, `perfect`, `completed`, `ongoing`.

Example:
```metta
(: f1 (ReadEvent e1) (STV 1.0 1.0))
(: f2 (Agent e1 she1) (STV 1.0 1.0))
(: f3 (Aspect e1 progressive) (STV 1.0 1.0))
```

9. Quantification
- Universal -> `Implication` with variables.
- Existential assertion -> witness constant.
- Existential question -> query variable.
- Cardinality -> `FoldAll` / `FoldAllValue` + `GreaterThan` if supported.

10. Negation
- Use `Not` over the proposition.
- Never treat absence of a fact as negation.
- For negative rules, place `Not` in the conclusion if the ontology uses explicit negative propositions.

Examples:
```metta
(: f1 (Not (Call alice)) (STV 1.0 1.0))
(: r1
   (Implication
      (Premises (Dog $x))
      (Conclusions (Not (Fly $x))))
   (STV 1.0 1.0))
```

11. Modality
- No built-in modal operators are documented.
- Use custom modal predicates such as `Can`, `May`, `Must`, `Possible`, `Necessary`, `Permitted`.
- Keep the modal separate from the action.

Examples:
```metta
(: f1 (Can alice swim) (STV 1.0 1.0))
(: f2 (May addressee1 leave) (STV 1.0 1.0))
(: f3 (Must addressee1 return) (STV 1.0 1.0))
```

If event structure is needed, attach the modal to an event constant instead.

12. Causation and purpose
- Reify both events and connect them with `Cause` or `Purpose`.

Example:
```metta
(: f1 (StudyEvent e1) (STV 1.0 1.0))
(: f2 (Agent e1 she1) (STV 1.0 1.0))
(: f3 (PassEvent e2) (STV 1.0 1.0))
(: f4 (Agent e2 she1) (STV 1.0 1.0))
(: f5 (Purpose e1 e2) (STV 1.0 1.0))
```

13. Comparison
- If a numeric scale is available, represent the measured value explicitly and use `GreaterThan`.
- Less-than is represented by reversed arguments.
- Equality is not built in; use same value facts or a custom relation such as `SameHeight` if needed.

Examples:
```metta
(: f1 (Height tom 190 centimeter) (STV 1.0 1.0))
(: f2 (Height sam 175 centimeter) (STV 1.0 1.0))
(: r1
   (Implication
      (Premises
         (Height $x $hx centimeter)
         (Height $y $hy centimeter)
         (GreaterThan $hx $hy))
      (Conclusions
         (TallerThan $x $y)))
   (STV 1.0 1.0))
```

For superlatives, prefer explicit score/value comparisons if available. Otherwise use a custom predicate like `Best option1`, but do not bake the compared set into the predicate name.

14. Conjunction and disjunction
- Conjunction in assertions -> emit multiple facts, or one rule with multiple premises/conclusions.
- Conjunction in noun phrases -> usually duplicate the predicate over each conjunct.
- No built-in disjunction operator is documented.

Examples:
“Anna sang and Ben danced.”
```metta
(: f1 (Sing anna) (STV 1.0 1.0))
(: f2 (Dance ben) (STV 1.0 1.0))
```

For disjunction:
- If English means both alternatives are permitted/available, assert both.
- If English means a true logical disjunction with unresolved branch, use a custom relation such as `Alternative` or leave to upstream handling; do not invent an undocumented `Or`.

15. Conditionals
- Use `Implication`.
- Variables in the rule are the mechanism for general conditional meaning.

Example:
```metta
(: r1
   (Implication
      (Premises (RainEvent e1))
      (Conclusions (StayHome we1)))
   (STV 1.0 1.0))
```

For ordinary generic conditionals over individuals:
```metta
(: r1
   (Implication
      (Premises (Tired $x))
      (Conclusions (Rest $x)))
   (STV 1.0 1.0))
```

Counterfactual distinctions are not natively marked; encode only if an approximation is acceptable.

16. Propositional attitudes and embedded clauses
- Reify the embedded event/state/proposition and link it with `Content`.
- Use attitude predicates such as `Believe`, `Know`, `Want`, `Hope`, or attitude-event decomposition.

Example:
```metta
(: f1 (LeaveEvent e1) (STV 1.0 1.0))
(: f2 (Agent e1 lee) (STV 1.0 1.0))
(: f3 (Want lee e1) (STV 1.0 1.0))
```

For more structure:
```metta
(: f1 (WantEvent w1) (STV 1.0 1.0))
(: f2 (Experiencer w1 lee) (STV 1.0 1.0))
(: f3 (Content w1 e1) (STV 1.0 1.0))
```

17. Adverbs and manner
- Attach adverbial information as a separate modifier on the event.
- Prefer a general `Manner` relation when the modifier value itself may be queried.

Example:
```metta
(: f1 (PackEvent e1) (STV 1.0 1.0))
(: f2 (Agent e1 she1) (STV 1.0 1.0))
(: f3 (Theme e1 box1) (STV 1.0 1.0))
(: f4 (Manner e1 carefully) (STV 1.0 1.0))
```

18. Frequency and habituality
- Represent frequency as a separate feature: `Frequency e1 often`, `Frequency e1 usually`.
- Mark habits with a custom predicate like `Habitual` when needed.

Example:
```metta
(: f1 (VisitEvent e1) (STV 1.0 1.0))
(: f2 (Agent e1 she1) (STV 1.0 1.0))
(: f3 (Frequency e1 often) (STV 1.0 1.0))
(: f4 (Time e1 sunday) (STV 1.0 1.0))
(: f5 (Habitual e1) (STV 1.0 1.0))
```

19. Pronouns, reference, and coreference
- Resolve pronouns to the same constant as their antecedent.
- Do not create a new entity for a coreferent pronoun.
- If reference is unknown, do not guess.

Example:
“Maria saw a dog and petted it.”
```metta
(: f1 (Dog dog1) (STV 1.0 1.0))
(: f2 (SeeEvent e1) (STV 1.0 1.0))
(: f3 (Agent e1 maria) (STV 1.0 1.0))
(: f4 (Patient e1 dog1) (STV 1.0 1.0))
(: f5 (PetEvent e2) (STV 1.0 1.0))
(: f6 (Agent e2 maria) (STV 1.0 1.0))
(: f7 (Patient e2 dog1) (STV 1.0 1.0))
```

20. Reflexives and reciprocals
- Reflexive: use the same entity constant in both roles.
- Reciprocal: either emit paired binary facts or one event with multiple participants plus `Reciprocal`.

Reflexive:
```metta
(: f1 (BlameEvent e1) (STV 1.0 1.0))
(: f2 (Agent e1 john) (STV 1.0 1.0))
(: f3 (Patient e1 john) (STV 1.0 1.0))
```

Reciprocal pairwise:
```metta
(: f1 (Hug anna beth) (STV 1.0 1.0))
(: f2 (Hug beth anna) (STV 1.0 1.0))
```

21. Passive voice and alternations
- Preserve underlying semantic roles; do not let surface subjecthood change role assignment.
- If the agent is omitted, omit it rather than inventing one.

Example:
“The cake was eaten by the children.”
```metta
(: f1 (EatEvent e1) (STV 1.0 1.0))
(: f2 (Patient e1 cake1) (STV 1.0 1.0))
(: f3 (Agent e1 children1) (STV 1.0 1.0))
```

“The door opened.”
```metta
(: f1 (OpenEvent e1) (STV 1.0 1.0))
(: f2 (Patient e1 door1) (STV 1.0 1.0))
```

22. Relative clauses and nominal modification
- Use one constant for the head noun.
- Encode the relative clause as additional facts about that same constant.

Example:
“The man who called is waiting.”
```metta
(: f1 (Man man1) (STV 1.0 1.0))
(: f2 (Call man1) (STV 1.0 1.0))
(: f3 (Wait man1) (STV 1.0 1.0))
```

If the clause needs richer structure, use event decomposition.

23. Questions and wh-phrases
- Use the query envelope `(: $prf pattern $tv)`.
- Yes/no question -> grounded pattern.
- Wh-question -> variable in the queried position.

Examples:
“Did Maria call?”
```metta
(: $prf (Call maria) $tv)
```

“Who called?”
```metta
(: $prf (Call $x) $tv)
```

“What did she buy?”
After pronoun resolution:
```metta
(: $prf (Buy she1 $x) $tv)
```

If the answer depends on a conjunction of facts, derive a summary predicate first and query that.

24. Imperatives and directives
- Imperatives are not ordinary world facts.
- If you need to represent them, model them as speech acts: `Command`, `Request`, `Prohibit`.

Example:
```metta
(: f1 (CloseEvent e1) (STV 1.0 1.0))
(: f2 (Patient e1 door1) (STV 1.0 1.0))
(: f3 (Command speaker_1 addressee_1 e1) (STV 1.0 1.0))
```

Negative imperative:
```metta
(: f1 (EnterEvent e1) (STV 1.0 1.0))
(: f2 (Agent e1 addressee_1) (STV 1.0 1.0))
(: f3 (Prohibit speaker_1 addressee_1 e1) (STV 1.0 1.0))
```

25. Generic and kind-level statements
- Instance-level generic that licenses inference -> rule.
- Kind-level property that applies to the species/type itself -> use a kind constant.

Examples:
“Dogs bark.”
```metta
(: r1
   (Implication
      (Premises (Dog $x))
      (Conclusions (Bark $x)))
   (STV 0.9 0.7))
```

“Tigers are endangered.” If the intention is kind-level:
```metta
(: f1 (Endangered tiger_kind) (STV 1.0 1.0))
```

26. Part-whole and membership relations
- Use atomic binary predicates such as `PartOf`, `MemberOf`, `BelongTo`.
- Choose the relation that matches the intended ontology; do not overload class membership with part-whole.

Examples:
```metta
(: f1 (PartOf wheel1 bike1) (STV 1.0 1.0))
(: f2 (MemberOf alaska united_states) (STV 1.0 1.0))
(: f3 (ChapterOf chapter1 book1) (STV 1.0 1.0))
```

27. Measurement, amounts, and numerals
- Put numbers in arguments, not in predicate names.
- Prefer a separate unit argument:
```metta
(: f1 (Weight bag1 2 kilogram) (STV 1.0 1.0))
(: f2 (Duration movie1 3 hour) (STV 1.0 1.0))
(: f3 (Height she1 180 centimeter) (STV 1.0 1.0))
```
- Compare numeric values using `GreaterThan`.

For uncertain measured values, use distribution-valued predicates:
```metta
(: f1 (WeightDist bag1 (ParticleFromNormal 2 0.1)) (STV 1.0 1.0))
```

28. Degree and intensification
- Keep the base property separate from degree.
- Use a separate `Degree` relation or an explicit numeric scale if available.

Example:
```metta
(: f1 (Hot coffee1) (STV 1.0 1.0))
(: f2 (Degree coffee1 hot very) (STV 1.0 1.0))
```

For scalar properties, prefer explicit values when possible:
```metta
(: f1 (Slope road1 18 degree) (STV 1.0 1.0))
```

29. Discourse sequencing and connectives
- Reify linked clauses/events and connect them with discourse relations such as `Before`, `Cause`, `Contrast`, `Explanation`.

Example:
```metta
(: f1 (EatEvent e1) (STV 1.0 1.0))
(: f2 (GoHomeEvent e2) (STV 1.0 1.0))
(: f3 (After e2 e1) (STV 1.0 1.0))
(: f4 (Cause tired1 e2) (STV 1.0 1.0))
```

Use event-to-event or proposition-to-proposition links rather than creating one large discourse predicate.

30. Distributive and collective readings
- Distributive -> separate facts/events for each participant when the members are known.
- Collective -> one event with multiple participants, optionally marked by `Collective`.

Examples:
Distributive:
```metta
(: f1 (Receive anna toy1) (STV 1.0 1.0))
(: f2 (Receive ben toy2) (STV 1.0 1.0))
```

Collective:
```metta
(: f1 (LiftEvent e1) (STV 1.0 1.0))
(: f2 (Agent e1 students1) (STV 1.0 1.0))
(: f3 (Patient e1 piano1) (STV 1.0 1.0))
(: f4 (Collective e1) (STV 1.0 1.0))
```

31. Scope interactions
- Scope is expressed only indirectly through rule structure, witness constants, variables, and `Not`.
- Be conservative: do not collapse distinct readings into the same encoding.

Examples:
- “A student did not leave” -> witness + `Not (Leave student1)`
- “Every student may leave” -> rule concluding a modal fact
```metta
(: r1
   (Implication
      (Premises (Student $x))
      (Conclusions (May $x leave)))
   (STV 1.0 1.0))
```

If an English scope reading cannot be represented without loss, prefer a marked approximation or upstream resolution.

32. Reported speech and quotation
- Represent a speech/report event and link it to content with `Content`.
- There is no documented raw quotation-string syntax; normalize quoted material into ordinary proposition/event structure.

Example:
```metta
(: f1 (Late train1) (STV 1.0 1.0))
(: f2 (SayEvent e1) (STV 1.0 1.0))
(: f3 (Speaker e1 she1) (STV 1.0 1.0))
(: f4 (Content e1 train1) (STV 1.0 1.0))
```

If content is more complex than one atom, reify it as its own event/state constant and attach `Content e1 p1`.

33. Transfer and ditransitive arguments
- Use either a direct ternary predicate for simple cases or event decomposition for richer structure.
- Recommended role schema: `Agent`, `Recipient`, `Theme`, optionally `Source`.

Example:
```metta
(: f1 (GiveEvent e1) (STV 1.0 1.0))
(: f2 (Agent e1 she1) (STV 1.0 1.0))
(: f3 (Recipient e1 him1) (STV 1.0 1.0))
(: f4 (Theme e1 book1) (STV 1.0 1.0))
```

34. Apposition and naming
- Use one entity constant.
- Assert all appositive descriptions of that same constant.
- Use `Name` / `Alias` when needed.

Example:
“Lee, the surgeon, arrived.”
```metta
(: f1 (Person lee1) (STV 1.0 1.0))
(: f2 (Surgeon lee1) (STV 1.0 1.0))
(: f3 (Arrive lee1) (STV 1.0 1.0))
(: f4 (Name lee1 lee) (STV 1.0 1.0))
```

35. Ellipsis and gapping
- Recover omitted material from context before emission.
- Emit the fully reconstructed semantics, not a partial skeleton with missing arguments.
- If recovery is uncertain, either omit the fact or lower confidence.

Example:
“Alice ordered tea and Bob coffee.”
```metta
(: f1 (Order alice tea1) (STV 1.0 1.0))
(: f2 (Order bob coffee1) (STV 1.0 1.0))
```

36. Clefts and focus
- Strip the cleft scaffold and encode the core proposition.
- If focus matters, add a separate focus relation.

Example:
“It was John who called.”
```metta
(: f1 (Call john) (STV 1.0 1.0))
(: f2 (Focus john) (STV 1.0 1.0))
```

For exclusives like “Only Maria laughed”:
- If the closed set is known, encode the positive fact plus explicit negative facts for alternatives.
- Otherwise use a custom relation such as `Only maria laugh`, keeping `Only` atomic and the property/action as an argument, not in the predicate name.

11. Final consistency checklist

Before emitting a parse, verify:
- The outer form is `(: ... ... ...)`.
- Statement proof ids are constants; query proof ids are variables.
- Predicate names are CamelCase, lemmatized, and atomic.
- Constants are stable, lowercase atoms.
- Numbers are arguments, not part of predicate names.
- Comparisons use `GreaterThan`.
- Negation uses `Not`.
- Universal meaning uses rules, not invented quantifier syntax.
- Existential assertions use witness constants.
- Complex clauses are decomposed into linked atoms.
- Queries ask only for the requested unknown, usually as one target pattern.
- No undocumented syntax or operators have been introduced.

---

## Relation templates

Canonical primitive relation templates for PeTTaChainer

Legend
- `<...>` = template slot, not a literal emitted atom
- `$x`, `$y` = entity variables
- `$e` = event variable
- `$s` = state variable
- `$p` = embedded event/state/proposition variable
- `$t` = time variable
- `$n` = numeric/count variable
- `$d` = distribution variable
- `$unit` = unit atom
- Open-class lexical families may instantiate many predicate heads.
- Closed-class relations below should use the exact predicate names shown.

======================================================================
1. PUBLIC LOGICAL / QUERY SCAFFOLDING
======================================================================

1.1 Fact / assertion envelope
```metta
(: <fact-id> <proposition> (STV <strength> <confidence>))
```
Use for asserted facts.

1.2 Rule envelope
```metta
(: <rule-id>
   (Implication
      (Premises
         <premise1>
         <premise2>
         ...)
      (Conclusions
         <conclusion1>
         <conclusion2>
         ...))
   (STV <strength> <confidence>))
```
Use for generics, taxonomic rules, conditionals, and derived summaries.

1.3 Query envelope
```metta
(: $prf <pattern> $tv)
```
Use for yes/no and wh-queries. Put variables only in the unknown positions.

======================================================================
2. BUILT-IN LOGICAL / COMPOSITIONAL OPERATORS
======================================================================

These are fixed built-ins, not custom ontology predicates.

2.1 Negation
```metta
(Not <proposition>)
```
Explicit negation with local scope.

2.2 Conditional / rule structure
```metta
(Implication (Premises <premise> ...) (Conclusions <conclusion> ...))
(Premises <premise1> <premise2> ...)
(Conclusions <conclusion1> <conclusion2> ...)
```
Canonical encoding for if-then structure, universal/generic rules, and derived relations.

2.3 Numeric comparison
```metta
(GreaterThan <left> <right>)
```
Canonical comparison primitive. Use reversed arguments for less-than meanings.

2.4 Runtime computation
```metta
(Compute <function> (<arg1> <arg2> ...) -> $out)
```
Use only when a runtime function already exists.

2.5 Aggregation / counting
```metta
(FoldAll <pattern> <value> <init> <fold-fn> -> $out)
(FoldAllValue <pattern> <init> <fold-fn> -> $out)
```
Use for counts, totals, and aggregate-derived quantificational facts.

2.6 Distribution mapping / aggregation
```metta
(MapDist <function> <pattern> $d -> $outD)
(Map2Dist <function> <patternA> $dA <patternB> $dB -> $outD)
(AverageDist <pattern> $d -> $outD)
```
Use for uncertain values stored as distributions.

======================================================================
3. TRUTH / VALUE CONSTRUCTORS
======================================================================

3.1 Truth uncertainty
```metta
(STV <strength> <confidence>)
```

3.2 Value uncertainty constructors
```metta
(NatDist ((<value> <probability>) ...))
(FloatDist ((<value> <probability>) ...))
(ParticleDist <ref>)
(ParticleDist <ref> <scale>)
(PointMass <value>)
(ParticleFromNormal <mu> <sigma>)
(ParticleFromPairs ((<value1> <weight1>) (<value2> <weight2>) ...))
```

These constructors appear inside propositions, typically in `...Dist` predicates.

======================================================================
4. OPEN-CLASS LEXICAL PREDICATE FAMILIES
======================================================================

These are the productive content-predicate families used across domains.

4.1 Entity / kind membership
```metta
(<Kind> $x)
```
Unary class/category membership.
Examples of the family: `Dog`, `City`, `Person`, `Doctor`.

4.2 One-place properties / states
```metta
(<Property> $x)
```
Unary qualities or states.
Examples of the family: `Red`, `Fast`, `Hot`, `Friendly`, `Endangered`.

4.3 Simple intransitive predicate family
```metta
(<IntransitivePredicate> $x)
```
Compact form for simple unmodified clauses when event reification is unnecessary.
Examples of the family: `Bark`, `Laugh`, `Arrive`, `Wait`.

4.4 Simple binary predicate family
```metta
(<BinaryPredicate> $x $y)
```
Compact form for simple binary relations/events when no extra roles or modifiers are needed.
Examples of the family: `Own`, `See`, `Read`, `Call`, `Hug`.

4.5 Simple ternary predicate family
```metta
(<TernaryPredicate> $x $y $z)
```
Compact form for simple ditransitives or three-place relations.
Examples of the family: `Give`, `Tell`, `Put`.

4.6 Reified event type family
```metta
(<VerbLemma>Event $e)
```
Preferred primitive for complex clauses, modifiers, embedded content, passive alternations, and role-based querying.
Examples of the family: `CutEvent`, `ReadEvent`, `GiveEvent`, `SayEvent`, `OpenEvent`.

4.7 Reified state/situation type family
```metta
(<StateLemma>State $s)
```
Use for embedded stative content, resultant states, or reified situations.
Examples of the family: `LateState`, `ReadyState`, `ClosedState`.

4.8 Explicit scalar/measurement family
```metta
(<ScalarProperty> $x $n $unit)
```
Numeric property with explicit unit.
Examples of the family: `Height`, `Weight`, `Duration`, `Length`, `Temperature`.

4.9 Uncertain scalar/measurement family
```metta
(<ScalarProperty>Dist $x $d)
```
Distribution-valued property.
Examples of the family: `HeightDist`, `WeightDist`, `AreaDist`.

Note
- Prefer 4.6/4.7 plus role relations once a clause has modifiers, tense/aspect, embedding, cause, purpose, negation scope, or multiple participants.
- Derived summary predicates such as `(Cut $a $p)` are useful, but they are not primitive templates.

======================================================================
5. CLOSED-CLASS SEMANTIC RELATION TEMPLATES
======================================================================

----------------------------------------------------------------------
5.1 Reference, naming, definiteness, and deixis
----------------------------------------------------------------------

```metta
(Name $x <name_atom>)
(Alias $x <alias_atom>)
(Definite $x)
(Indefinite $x)
(Specific $x)
(Proximal $x)
(Distal $x)
```

Descriptions
- `Name`: canonical name or surface name token mapped to an entity.
- `Alias`: alternate label or nickname.
- `Definite`: referent treated as discourse-familiar/resolved.
- `Indefinite`: referent introduced as new/non-familiar.
- `Specific`: speaker has a particular referent in mind.
- `Proximal`: this/these-style deixis.
- `Distal`: that/those-style deixis.

----------------------------------------------------------------------
5.2 Event/state participant and modifier roles
----------------------------------------------------------------------

```metta
(Agent $e $x)
(Patient $e $x)
(Theme $e $x)
(Recipient $e $x)
(Beneficiary $e $x)
(Instrument $e $x)
(Experiencer $e $x)
(Stimulus $e $x)
(Source $e $x)
(Goal $e $x)
(Path $e $x)
(Location $e $x)
(Time $e $t)
(StartTime $e $t)
(EndTime $e $t)
(Manner $e <manner_atom>)
(Frequency $e <frequency_atom>)
(Tense $e <tense_atom>)
(Aspect $e <aspect_atom>)
(Content $e $p)
(Cause $x $y)
(Purpose $e $p)
(Reason $e $p)
(Result $e $s)
(Speaker $e $x)
(Addressee $e $x)
(Participant $e $x)
```

Descriptions
- `Agent`: actor, doer, or effective causer of an event.
- `Patient`: entity directly affected or undergone.
- `Theme`: moved, transferred, perceived, or semantically central participant.
- `Recipient`: receiver in transfer/communication.
- `Beneficiary`: one for whose benefit the event occurs.
- `Instrument`: means/tool used in the event.
- `Experiencer`: experiencer of mental/perceptual/emotional state.
- `Stimulus`: entity causing the experience or emotion.
- `Source`: origin in motion, transfer, or derivation.
- `Goal`: destination/end-point in motion, transfer, or change.
- `Path`: route traversed.
- `Location`: place of an event/state, or anchoring location.
- `Time`: general temporal anchor.
- `StartTime`: event/state onset.
- `EndTime`: event/state endpoint.
- `Manner`: how the event happens.
- `Frequency`: often/usually/daily-style recurrence marker.
- `Tense`: typical values include `past`, `present`, `future`.
- `Aspect`: typical values include `progressive`, `perfect`, `completed`, `ongoing`.
- `Content`: embedded clause/event/state content.
- `Cause`: general causal link; may relate event-event, state-event, or proposition-proposition.
- `Purpose`: intended goal of an event.
- `Reason`: explanatory/motivational reason.
- `Result`: resulting state/situation produced by an event.
- `Speaker`: producer of a speech/report event.
- `Addressee`: intended hearer/recipient of a speech/directive event.
- `Participant`: additional participant, useful for collective or reciprocal structures.

----------------------------------------------------------------------
5.3 Possession, ownership, part-whole, and membership
----------------------------------------------------------------------

```metta
(Have $x $y)
(Own $x $y)
(BelongTo $x $y)
(PartOf $x $y)
(MemberOf $x $y)
```

Descriptions
- `Have`: general possession/holding/association.
- `Own`: stronger ownership relation.
- `BelongTo`: inverse-style belonging/affiliation.
- `PartOf`: component-whole relation.
- `MemberOf`: member-group / element-collection relation.

----------------------------------------------------------------------
5.4 Spatial configuration and motion-related relations
----------------------------------------------------------------------

These may relate entity-entity, entity-place, or event-landmark pairs as needed.

```metta
(In $x $y)
(On $x $y)
(At $x $y)
(Under $x $y)
(Over $x $y)
(Above $x $y)
(Below $x $y)
(Beside $x $y)
(Near $x $y)
(Inside $x $y)
(Outside $x $y)
(Between $x $y)
(Through $x $y)
(Across $x $y)
(Around $x $y)
(Toward $x $y)
(Behind $x $y)
(FrontOf $x $y)
```

Descriptions
- `In`, `On`, `At`: basic locative relations.
- `Under`, `Over`, `Above`, `Below`: vertical/topological relations.
- `Beside`, `Near`, `Between`: proximity/adjacency relations.
- `Inside`, `Outside`: interior/exterior relation.
- `Through`, `Across`, `Around`, `Toward`: path/orientation relations, often with an event as first argument.
- `Behind`, `FrontOf`: frontal orientation relations.

----------------------------------------------------------------------
5.5 Temporal ordering and discourse linkage
----------------------------------------------------------------------

```metta
(Before $x $y)
(After $x $y)
(During $x $y)
(Simultaneous $x $y)
(Contrast $x $y)
(Explanation $x $y)
```

Descriptions
- `Before`, `After`: temporal ordering between events, states, or times.
- `During`: containment/overlap of one interval/event within another.
- `Simultaneous`: co-temporality.
- `Contrast`: discourse contrast/concession-style link.
- `Explanation`: explanatory relation between two reified clauses/events/states.

Note
- Use `Implication` as the canonical template for true conditionals and generic if-then statements.
- Use `Before` / `After` rather than inventing tense morphology in predicate names.

----------------------------------------------------------------------
5.6 Modality
----------------------------------------------------------------------

```metta
(Can $x $p)
(May $x $p)
(Must $x $p)
(Possible $p)
(Necessary $p)
(Permitted $x $p)
```

Descriptions
- `Can`: ability/capacity of an agent with respect to content.
- `May`: permission/possibility licensed for an agent.
- `Must`: obligation/necessity on an agent.
- `Possible`: proposition/event is possible.
- `Necessary`: proposition/event is necessary.
- `Permitted`: explicit permission relation when separate from `May` is useful.

`$p` should usually be a reified event/state/proposition constant.

----------------------------------------------------------------------
5.7 Propositional attitudes and intention
----------------------------------------------------------------------

```metta
(Believe $x $p)
(Know $x $p)
(Want $x $p)
(Hope $x $p)
(Intend $x $p)
```

Descriptions
- `Believe`: belief attitude toward content.
- `Know`: knowledge attitude toward content.
- `Want`: desire toward content.
- `Hope`: hope toward content.
- `Intend`: intention/plan toward content.

Use with a reified event/state/proposition in `$p`, or decompose further with `(<AttitudeVerb>Event $e)` plus `Experiencer` and `Content`.

----------------------------------------------------------------------
5.8 Directives and speech-act force
----------------------------------------------------------------------

```metta
(Command $x $y $p)
(Request $x $y $p)
(Prohibit $x $y $p)
```

Descriptions
- `Command`: speaker `$x` commands addressee `$y` regarding content `$p`.
- `Request`: speaker `$x` requests addressee `$y` regarding content `$p`.
- `Prohibit`: speaker `$x` forbids addressee `$y` regarding content `$p`.

These cover imperatives, prohibitions, and directive readings.

----------------------------------------------------------------------
5.9 Quantity, counting, grouping, and reading type
----------------------------------------------------------------------

```metta
(Cardinality $x $n)
(Quantity $x $n $unit)
(Collective $e)
(Distributive $e)
(Reciprocal $e)
(Habitual $e)
```

Descriptions
- `Cardinality`: explicit count of a group/collection.
- `Quantity`: measured amount of an entity/substance.
- `Collective`: event interpreted as joint group action.
- `Distributive`: event interpreted as individually distributed over participants.
- `Reciprocal`: event interpreted mutually among participants.
- `Habitual`: event/schema interpreted as habitual or regular.

Note
- Exact numeric quantification can also be derived with `FoldAll` / `FoldAllValue` plus `GreaterThan` and `Not`.

----------------------------------------------------------------------
5.10 Degree, equality, comparison, and ranking
----------------------------------------------------------------------

```metta
(Degree $x <scale_atom> <degree_atom>)
(Equal $x $y)
(Maximal $x <scale_atom> $context)
(Minimal $x <scale_atom> $context)
```

Descriptions
- `Degree`: intensification or attenuation on a property/manner/scale.
- `Equal`: explicit asserted equality/equivalence when needed as a custom relation.
- `Maximal`: superlative/top-ranked item on a given scale within a context/comparison class.
- `Minimal`: bottom-ranked item on a given scale within a context/comparison class.

Notes
- Prefer explicit scalar facts plus `GreaterThan` for ordinary comparatives.
- Use reversed `GreaterThan` for less-than meanings.
- `Equal` is a custom predicate, not a built-in arithmetic operator.

----------------------------------------------------------------------
5.11 Focus and alternatives
----------------------------------------------------------------------

```metta
(Focus $p $x)
(Only $x $p)
(Alternative $alt $option)
```

Descriptions
- `Focus`: content/event `$p` focuses constituent `$x`.
- `Only`: exclusive reading; only `$x` satisfies the focused slot in content `$p`.
- `Alternative`: reified alternative-set/choice/disjunction structure; `$alt` has option `$option`.

Note
- There is no built-in `Or`. `Alternative` is the canonical fallback when unresolved alternatives must be stored.

======================================================================
6. DISTRIBUTION-VALUED PROPOSITION TEMPLATES
======================================================================

Use these when the proposition is true but the value is uncertain.

6.1 Uncertain scalar property
```metta
(<ScalarProperty>Dist $x $d)
```

6.2 Example shape of the distribution argument
```metta
(<ScalarProperty>Dist $x (ParticleFromNormal <mu> <sigma>))
(<ScalarProperty>Dist $x (FloatDist ((<value1> <prob1>) (<value2> <prob2>) ...)))
(<ScalarProperty>Dist $x (PointMass <value>))
```

6.3 Distribution-derived summaries in rules
```metta
(MapDist <function> (<ScalarProperty>Dist $x $d) $d -> $outD)
(Map2Dist <function> (<ScalarProperty>Dist $x $d1) $d1 (<ScalarProperty>Dist $y $d2) $d2 -> $outD)
(AverageDist (<ScalarProperty>Dist $x $d) $d -> $outD)
```

======================================================================
7. PHENOMENA HANDLED BY COMPOSITION, NOT BY EXTRA PREDICATE HEADS
======================================================================

The following do not require additional primitive relation names:

7.1 Universal quantification / generics
- Use variables inside `Implication`.
```metta
(: <rule-id>
   (Implication
      (Premises (<Kind> $x))
      (Conclusions (<Property> $x)))
   (STV 1.0 1.0))
```

7.2 Existential quantification
- Use witness constants in assertions.
- Use variables in queries.

7.3 Conjunction
- Use multiple facts, multiple premises, or multiple conclusions.
- Do not invent `And`.

7.4 Negation scope
- Use nested `Not`.
- Do not treat missing facts as negative facts.

7.5 Questions and wh-phrases
- Reuse any proposition template inside:
```metta
(: $prf <pattern> $tv)
```

7.6 Pronouns, coreference, reflexives
- Reuse the same constant for the same referent.
- Reflexives are ordinary repeated-role assignments.
- No dedicated coreference predicate is required.

7.7 Passive voice
- Keep the same semantic roles; do not add a passive-specific primitive.

7.8 Relative clauses and apposition
- Reuse the same entity constant and add more facts about it.
- No dedicated relative-clause primitive is required.

7.9 Ellipsis and gapping
- Recover omitted material upstream, then emit full ordinary predicates.

======================================================================
8. CANONICAL PRIORITY ORDER
======================================================================

When multiple encodings are possible, prefer this order:

1. Built-in logical operators:
   `Implication`, `Premises`, `Conclusions`, `Not`, `GreaterThan`, `Compute`, `FoldAll`, `FoldAllValue`, `MapDist`, `Map2Dist`, `AverageDist`

2. Reified event/state families plus fixed role relations:
   `(<VerbLemma>Event $e)`, `(<StateLemma>State $s)`, plus `Agent`, `Patient`, `Theme`, etc.

3. Open-class kind/property predicates:
   `(<Kind> $x)`, `(<Property> $x)`

4. Compact direct lexical predicates only for simple, unmodified clauses:
   `(<IntransitivePredicate> $x)`, `(<BinaryPredicate> $x $y)`, `(<TernaryPredicate> $x $y $z)`

5. Derived summary predicates only as secondary query aids, not as primitives

======================================================================
9. MINIMAL CORE INVENTORY TO SHARE ACROSS THE PIPELINE
======================================================================

If a smaller mandatory closed-class inventory is needed, this is the core set:

```metta
Name Alias Definite Indefinite Specific Proximal Distal
Agent Patient Theme Recipient Beneficiary Instrument Experiencer Stimulus
Source Goal Path Location Time StartTime EndTime Manner Frequency Tense Aspect
Content Cause Purpose Reason Result Speaker Addressee Participant
Have Own BelongTo PartOf MemberOf
In On At Under Over Above Below Beside Near Inside Outside Between Through Across Around Toward Behind FrontOf
Before After During Simultaneous Contrast Explanation
Can May Must Possible Necessary Permitted
Believe Know Want Hope Intend
Command Request Prohibit
Cardinality Quantity Collective Distributive Reciprocal Habitual
Degree Equal Maximal Minimal
Focus Only Alternative
```

This closed-class inventory, combined with the open-class lexical families and the built-ins above, is sufficient to cover the listed linguistic phenomena.