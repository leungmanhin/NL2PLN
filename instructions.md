# NL to logic conversion instructions

You are converting English natural language (sentences or questions) into logical expressions (statements or queries) for a chainer/reasoner.  The chainer's syntax and semantics reference is provided separately as the `pln_spec` input (see `chainer_analysis.txt`).

## Chainer primitives
*Quick-reference cheat sheet of names with semantic backing.  Use these names verbatim wherever a primitive is needed.*

PeTTaChainer primitive quick reference

Public bare statement and query scaffolding
  `(: <proof-id> <proposition> <tv>)` — outer envelope for stored facts and stored rules.
  `(: <rule-id> (Implication ...) (STV strength confidence))` — rule stored as a statement.
  `(: $prf <pattern> $tv)` — query pattern; proof-id position must be a variable.
  `$name` — variable syntax used in facts, rules, and queries.
  `->` — output-binder marker used inside compute, fold, and distribution helper forms.

Rule and logical forms
  `(Implication (Premises ...) (Conclusions ...))` — if-then rule proposition.
  `(Premises premise1 premise2 ...)` — antecedent list inside an implication.
  `(Conclusions conclusion1 conclusion2 ...)` — consequent list inside an implication.
  `(Not expr)` — negation / negative premise over the nested expression.
  `(Compute f (arg1 arg2 ...) -> $out)` — compute a function result and bind it.
  `(FoldAll pattern value init fold-fn -> out)` — aggregate over matching facts with an explicit value.
  `(FoldAllValue pattern init fold-fn -> out)` — aggregate over matching fact values.
  `(GreaterThan a b)` — documented numeric/distribution greater-than comparison.
  `(> a b)` — documented alias/sugar for the greater-than comparison family.

Distribution helper forms
  `(MapDist f (DistFactA ... $inDist) $inDist -> $outDist)` — map a function over a distribution-valued fact.
  `(Map2Dist f (DistFactA ... $distA) $distA (DistFactB ... $distB) $distB -> $outDist)` — combine two distribution-valued facts.
  `(AverageDist (DistFactPattern ... $inDist) $inDist -> $outDist)` — average over matching distribution-valued facts.

Lower-level / compiled distribution structural forms
  `(DistGreaterThanFormula ...)` — supported structural distribution comparison form.
  `(DistGreaterThanDistFormula ...)` — supported structural distribution-vs-distribution comparison form.
  `(ParticleMap ...)` — supported particle-map structural form.
  `(ParticleMap2 ...)` — supported two-input particle-map structural form.
  `(ParticleAddBernoulliFromSTV ...)` — supported particle/STV bridge structural form.

Truth-value and distribution constructors
  `(STV strength confidence)` — truth uncertainty for a proposition.
  `(NatDist ((value probability) ...))` — exact discrete natural-number distribution.
  `(FloatDist ((value probability) ...))` — exact discrete floating-point distribution.
  `(ParticleDist <ref>)` — particle distribution by opaque particle-store reference.
  `(ParticleDist <ref> <scale>)` — scaled particle distribution reference.
  `(PointMass x)` — degenerate distribution concentrated at one value.
  `(ParticleFromNormal mu sigma)` — particle distribution constructor from normal parameters.
  `(ParticleFromPairs ((x1 w1) (x2 w2) ...))` — particle distribution constructor from weighted pairs.

Utility / particle-store operators
  `(ParticleStoreCount ...)` — documented particle-store utility operator.
  `(ParticleStoreClear ...)` — documented particle-store clearing operator.
  `(ParticleStorePruneKB ...)` — documented particle-store / KB pruning operator.

Reserved internal runtime wrappers
  `!(eval <bare-expression>)` — internal evaluation wrapper; do not emit as external API input.
  `!(compileadd <kb> <evaluated-statement>)` — internal add wrapper; do not emit as external API input.
  `!(query <steps> <kb> <evaluated-query>)` — internal query wrapper; do not emit as external API input.

---

## Suggested vocabulary
*Naming suggestions for non-primitive concepts (thematic roles, spatial/temporal relations, etc.).  Use for consistency across translations; these names do NOT have chainer semantics unless explicit rules are added to the KB.*

IMPORTANT: The names below are naming suggestions for domain predicates and canonical relations only. They are not PeTTaChainer built-ins and have no built-in semantics unless the KB supplies facts or rules for them. Do not treat these as chainer primitives.

Open-class predicate template families
  `(ClassName entity)` — unary class/kind/category membership.
  `(EventType event)` — event occurrence/type named by a lemmatized verb or event noun.
  `(PropertyName entity)` — qualitative property or adjective-like attribute.
  `(StateName bearer)` — state or condition of an entity or event.
  `(DimensionUnit bearer value)` — measured value with dimension and unit in the predicate name.
  `(DimensionUnitDist bearer dist)` — uncertain measured value whose value argument is a distribution.
  `(CountName bearer count)` — count or cardinality value exposed as an argument.
  `(RelationName arg1 arg2 ...)` — reusable domain-specific relation.
  `(ContentType content)` — reified embedded proposition, question, command, quote, or action content.

Entity naming, reference, and discourse status
  `(Name entity nameId)` `(Alias entity nameId)` `(Mention mention)` `(RefersTo mention entity)` `(Corefers mention1 mention2)` `(Appositive entity descriptor)` `(Definite mention)` `(Indefinite mention)` `(Demonstrative mention)` `(Specific mention)` `(DiscourseNew mention)` `(DiscourseOld mention)` `(Speaker context entity)` `(Addressee context entity)`
  Disambiguation notes:
  - Name vs Alias: use `Name` for the primary recorded name and `Alias` for alternate names.
  - RefersTo vs Corefers: `RefersTo` links a mention to an entity; `Corefers` links two mentions to each other.
  - Speaker/Addressee: use for deictic or discourse context, not necessarily for every communication event.

Thematic and event roles
  `(Agent event participant)` `(Patient event participant)` `(Theme event participant)` `(Experiencer event participant)` `(Stimulus event participantOrContent)` `(Instrument event entity)` `(Recipient event participant)` `(Beneficiary event participant)` `(Source event entity)` `(Destination event entity)` `(Location event place)` `(Path event pathOrPlace)` `(Manner event manner)` `(Topic event topic)` `(Result event outcome)`
  Disambiguation notes:
  - Agent vs Cause: `Agent` is an event participant acting intentionally or causally; `Cause` relates two events/states or a cause to an effect.
  - Patient vs Theme: `Patient` is affected or changed; `Theme` is moved, transferred, perceived, or discussed without necessarily being changed.
  - Recipient vs Beneficiary: `Recipient` receives a transferred theme; `Beneficiary` benefits from the event.
  - Source vs Destination: `Source` is the origin; `Destination` is the endpoint.

Possession, part-whole, and group membership
  `(Owns owner owned)` `(Possesses holder held)` `(CustodyOf holder item)` `(BelongsTo item ownerOrWhole)` `(AssociatedWith entity1 entity2)` `(PartOf part whole)` `(HasPart whole part)` `(ComponentOf component system)` `(MemberOf member group)` `(HasMember group member)` `(Contains container content)`
  Disambiguation notes:
  - Owns vs Possesses: `Owns` is legal/social ownership; `Possesses` is current holding or control.
  - PartOf vs MemberOf: `PartOf` is component-whole structure; `MemberOf` is membership in a group or collection.
  - Contains vs HasPart: `Contains` is containment; `HasPart` is structural composition.

Spatial relations
  `(At entity place)` `(In entity place)` `(On entity support)` `(Under entity relatum)` `(Above entity relatum)` `(Below entity relatum)` `(Beside entity relatum)` `(AdjacentTo entity relatum)` `(Near entity relatum)` `(FarFrom entity relatum)` `(Inside entity container)` `(Outside entity relatum)` `(InFrontOf entity relatum)` `(Behind entity relatum)` `(Between entity relatum1 relatum2)` `(Over entity relatum)` `(Across entity region)` `(Through entity region)` `(Around entity relatum)`
  Disambiguation notes:
  - In vs Inside: `In` can be broad locative inclusion; `Inside` emphasizes interior containment.
  - On vs Above: `On` implies support/contact; `Above` does not.
  - Under vs Below: `Under` often implies coverage or vertical relation with a relatum; `Below` is purely lower position.
  - Across vs Through: `Across` crosses a surface/region; `Through` traverses an interior or passage.

Temporal relations and schedules
  `(Time eventuality time)` `(AtTime eventuality time)` `(Date eventuality date)` `(Before eventuality1 eventuality2OrTime)` `(After eventuality1 eventuality2OrTime)` `(During eventuality interval)` `(Overlaps intervalOrEvent1 intervalOrEvent2)` `(StartsAt eventuality time)` `(EndsAt eventuality time)` `(Until eventuality time)` `(Since eventuality time)` `(Duration eventualityOrEntity duration)` `(RecursOn eventOrPattern schedule)` `(Frequency eventOrPattern frequency)`
  Disambiguation notes:
  - Time vs StartsAt/EndsAt: `Time` is a general temporal attachment; `StartsAt` and `EndsAt` mark boundaries.
  - Before vs After: choose the predicate matching the stated ordering direction.
  - During vs Overlaps: `During` implies containment in an interval; `Overlaps` only implies shared temporal extent.
  - Frequency vs RecursOn: `Frequency` records how often; `RecursOn` records the schedule or calendar pattern.

Aspect, event status, and participation reading
  `(Ongoing event)` `(InProgress event)` `(Completed event)` `(Habitual eventOrPattern)` `(Repeated eventOrPattern)` `(Iterative eventOrPattern)` `(Scheduled event)` `(Planned event)` `(Canceled event)` `(Attempted event)` `(Collective event)` `(Distributive eventOrRelation)`
  Disambiguation notes:
  - Ongoing/InProgress vs Habitual: ongoing marks a current event; habitual marks a regular pattern.
  - Completed vs Scheduled: completed means realized; scheduled means planned for a time.
  - Attempted vs Completed: attempted does not entail success.
  - Collective vs Distributive: collective marks joint participation; distributive marks member-by-member participation.

Quantities, counts, and ordering values
  `(Quantity entity value)` `(Amount substanceOrEntity value)` `(Cardinality set count)` `(MemberCount group count)` `(EventCount eventPattern count)` `(Ordinal entity number)` `(Rank entity rank)` `(Total collection value)` `(Portion part whole fraction)`
  Disambiguation notes:
  - Quantity vs Amount: `Quantity` is general; `Amount` is preferred for masses, substances, or scalar extents.
  - Cardinality vs MemberCount: `Cardinality` can apply to any set-like object; `MemberCount` is specifically group membership.
  - Ordinal vs Rank: `Ordinal` is sequence position; `Rank` is ordered evaluation by a criterion.

Degree, intensity, and scalar qualification
  `(Degree bearer degree)` `(Intensity bearer level)` `(Scale dimension scale)` `(Threshold dimensionOrProperty value)` `(Modifier bearer modifier)` `(Diminished bearer level)` `(Intensified bearer level)`
  Disambiguation notes:
  - Degree vs Intensity: `Degree` is a scalar value on a property; `Intensity` is strength of a state/property.
  - Threshold vs Degree: `Threshold` is a cutoff; `Degree` is the observed or asserted level.

Comparison and evaluation
  `(SameValue entity1 entity2 dimension)` `(SameDegree entity1 entity2 dimension)` `(Equivalent entity1 entity2 criterion)` `(RankedAbove entity1 entity2 criterion)` `(RankedBelow entity1 entity2 criterion)` `(BetterThan entity1 entity2 criterion)` `(WorseThan entity1 entity2 criterion)` `(Best entity contextOrCriterion)` `(Worst entity contextOrCriterion)`
  Disambiguation notes:
  - SameValue vs SameDegree: `SameValue` is exact value identity; `SameDegree` is qualitative/scalar equality.
  - BetterThan vs RankedAbove: `BetterThan` is evaluative; `RankedAbove` is ordering by an explicit ranking criterion.
  - Best vs BetterThan: `Best` is superlative within a context; `BetterThan` compares two entities.

Modality, norms, and directives
  `(Able agent actionContent)` `(Possible content)` `(Necessary content)` `(Permitted agent actionContent)` `(Obligated agent actionContent)` `(Required content)` `(Prohibited agent actionContent)` `(Directive directiveEvent)` `(Command directiveEvent)` `(Request directiveEvent)` `(TargetAction directiveEvent actionContent)`
  Disambiguation notes:
  - Possible vs Permitted: `Possible` is circumstantial or epistemic; `Permitted` is deontic permission.
  - Necessary vs Obligated: `Necessary` applies to content generally; `Obligated` targets an agent.
  - Prohibited vs Not: `Prohibited` records a norm against an action, not the non-occurrence of the action.
  - Command vs Request: `Command` is stronger/authoritative; `Request` is weaker or polite.

Propositional attitudes, intentions, and embedded content
  `(Believe holder content)` `(Know holder content)` `(Doubt holder content)` `(Want holder content)` `(Desire holder content)` `(Hope holder content)` `(Intend holder content)` `(Expect holder content)` `(Fear holder content)` `(Prefer holder contentOrOption)` `(Content bearer content)` `(Proposition content)` `(QuestionContent content)` `(ActionContent content)`
  Disambiguation notes:
  - Know vs Believe: `Know` is factive if the ontology treats it that way; `Believe` need not be true.
  - Want/Desire vs Intend: wanting is preference; intending includes commitment toward action.
  - Hope vs Expect: hope is desire-oriented; expect is belief-oriented.
  - Proposition vs ActionContent: `Proposition` is truth-evaluable content; `ActionContent` is an action description.

Communication, reported speech, and quotation
  `(Say event)` `(Tell event)` `(Ask event)` `(Write event)` `(Report event)` `(Announce event)` `(Claim event)` `(Quote event)` `(Speaker event participant)` `(Addressee event participant)` `(Message event message)` `(Medium event medium)` `(Language event language)`
  Disambiguation notes:
  - Say vs Tell: `Tell` normally has an addressee; `Say` may not.
  - Report vs Claim: `Report` presents information as reported; `Claim` emphasizes asserted commitment.
  - Quote vs Message: `Quote` is quoted content as an object; `Message` is the communicated item.

Causation, purpose, and explanation
  `(Cause cause effect)` `(CausedBy effect cause)` `(Enables condition eventuality)` `(Prevents condition eventuality)` `(Purpose eventuality goalContent)` `(Goal bearer goalContent)` `(Reason eventuality reason)` `(Motivation agent reason)` `(Outcome event outcome)` `(Consequence cause effect)`
  Disambiguation notes:
  - Cause vs Reason: `Cause` is causal production; `Reason` is explanatory or justificatory.
  - Purpose vs Outcome: `Purpose` is intended; `Outcome` is what resulted.
  - Goal vs Purpose: `Goal` can belong to an agent or plan; `Purpose` modifies an event/action.
  - Prevents vs Prohibited: `Prevents` is causal blocking; `Prohibited` is normative banning.

Hypotheticality and factual status
  `(Hypothetical content)` `(Counterfactual content)` `(Presupposed content)` `(Asserted content)` `(Factual content)` `(Uncertain content)`
  Disambiguation notes:
  - Hypothetical vs Counterfactual: counterfactual implies contrary-to-fact status; hypothetical need not.
  - Presupposed vs Asserted: presupposed content is backgrounded; asserted content is directly put forward.
  - Uncertain vs Possible: `Uncertain` marks epistemic status of content; `Possible` marks modal possibility.

Alternatives, focus, and exclusivity
  `(Alternative context option)` `(Option context option)` `(Choice agent option)` `(MutuallyExclusive option1 option2)` `(Compatible option1 option2)` `(Focus content entityOrConstituent)` `(Only focus context)` `(Exclusive focus context)`
  Disambiguation notes:
  - Alternative vs Option: `Alternative` groups options under a contrast; `Option` is any available choice in a context.
  - Only vs Exclusive: `Only` represents focus-sensitive exclusivity; `Exclusive` is a more general exclusion relation.
  - MutuallyExclusive vs Compatible: mutually exclusive options cannot both hold; compatible options can.

Discourse relations
  `(Sequence segmentOrEvent1 segmentOrEvent2)` `(Contrast segment1 segment2)` `(Concession segment1 segment2)` `(Explanation segment1 segment2)` `(Elaboration segment1 segment2)` `(Background segment1 segment2)` `(Continuation segment1 segment2)` `(TopicShift segment1 segment2)`
  Disambiguation notes:
  - Sequence vs Before: `Sequence` is discourse/order structure; `Before` is temporal ordering.
  - Contrast vs Concession: contrast marks opposition; concession marks an unexpected coexistence.
  - Explanation vs Cause: `Explanation` relates discourse segments; `Cause` relates events, states, or facts.

Reflexive, reciprocal, and co-participation relations
  `(Reflexive event participant)` `(Reciprocal event groupOrParticipants)` `(Mutual event)` `(CoParticipant event participant)`
  Disambiguation notes:
  - Reflexive vs Reciprocal: reflexive links an argument back to itself; reciprocal marks mutual relations among participants.
  - Reciprocal vs Mutual: reciprocal emphasizes pairwise role reversal; mutual is a broader joint/mutual marker.

---

## Conversion guidelines
*Mapping rules from English to the chainer's logic.  Apply these patterns; cross-reference the vocabulary above for canonical names.*

# Conversion guidelines for English → PeTTaChainer logic

## 1. General conversion discipline

Convert English into small, compositional facts, rules, and queries. Each added statement should be a bare `(: ...)` expression. Do not emit internal wrappers. Use rules only for conditional, universal, generic, or law-like content. Use queries only to ask for derivable propositions.

Prefer decomposition over large predicate names. The chainer proves by matching predicate structure, variables, and built-in operators; if the reasoning is hidden inside a predicate string, it cannot participate compositionally in proofs.

---

## 2. Naming conventions

### Predicates and relations

Use CamelCase predicate names derived from lemmatized English:

- nouns/classes: `Dog`, `City`, `Doctor`, `Vehicle`
- verbs/events: `Call`, `Arrive`, `Build`, `Give`
- relations/roles: `Agent`, `Patient`, `Recipient`, `Instrument`, `Location`
- attributes: `Red`, `Friendly`, `Noisy`, or value relations such as `HeightCm`, `WeightKg`

Normalize away tense, plurality, and inflection:

- “dogs” → `Dog`
- “barked”, “barking” → `Bark`
- “was opened” → event predicate `Open` plus roles/aspect if needed

Do not encode negation, modality, tense, degree, quantities, thresholds, or named-entity combinations inside predicate names.

### Constants/entities

Use stable lowerCamelCase or mixed-case atoms:

- `alice`, `bob`, `fido`
- `room1`, `report9`, `rectA`
- `parisFrance`, `parisPerson1` when disambiguation is needed

Avoid natural-language strings; there is no quoted-string convention. For names or aliases, use ordinary atoms as name identifiers through domain predicates such as `Name` or `Alias`.

### Variables

Use `$`-prefixed variables.

Recommended conventions:

- `$x`, `$y`, `$z` for generic entities
- `$e` for events
- `$t` for times
- `$v`, `$n`, `$w`, `$dist` for values/distributions
- `$prf` for query proof id
- `$tv` for query truth value

Use readable variables in complex rules, but keep them short.

### Proof/rule ids

Use non-variable opaque ids for statements and rules. Use variables only in query proof-id position.

---

## 3. Entity representation and identity

The chainer has only symbol identity.

Therefore:

- same atom spelling = same entity
- different atom spelling = different entity
- there is no built-in `SameAs`, canonicalization, or name-resolution layer

For named entities, choose one canonical constant per real-world entity. If two entities share a surface name, create distinct constants and optionally add `Name` or `Alias` facts.

For apposition and naming:

- “Lee, the surgeon” should use one entity constant with both a name/alias relation and a `Surgeon` classification.
- “New York City, the Big Apple” should use one city constant with an alias relation.

For first- and second-person pronouns:

- resolve “I”, “we”, “you” from external context before conversion
- if context supplies a speaker/addressee constant, use it
- if no referent is known, do not invent a global constant like `i` or `you`
- for questions, leave the referent as a query variable only when the English question itself asks for that referent

For plural pronouns or groups, use a group constant plus `MemberOf` / `HasMember` style relations if individual members matter.

---

## 4. Quantification and scope

### Existential statements

Indefinites such as “a”, “some”, and “there is” usually introduce witness constants when asserted as facts.

Example policy in prose:

- “A dog barked” introduces a dog constant and a bark event constant.
- “Some students laughed” introduces one or more student witnesses unless the exact students are already known.

Do not represent existentiality by creating predicates such as `SomeStudentLaughed`.

### Existential queries

Use variables in the queried proposition.

For “Who called?”, query the relevant role or event proposition with a variable in the unknown position.

### Universal and generic statements

Use rules with variables.

Map:

- “Every X is Y”
- “All X do Y”
- “If X then Y”
- generic truths such as “Dogs bark”

to an `Implication` whose premises restrict the variable and whose conclusions state what follows.

Variables in rules carry the universal/generic force. Do not use nonexistent quantifier operators.

### Cardinality

For “N X exist/did Y”, introduce N witness constants when the sentence asserts concrete participants.

For count-valued claims, expose the number as an argument of a count predicate such as `MemberCount`, `EventCount`, or another domain-specific count relation. Do not place the number in the predicate name.

For derived counts, use aggregation machinery such as `FoldAll` / `FoldAllValue` where the ontology supplies the fold function. Use `GreaterThan` for documented greater-than tests. Since no public equality or less-than primitive is documented, exact and at-most constraints require either:

- an explicitly asserted count fact, or
- a domain-provided compute/fold convention that produces the needed exactness test

Do not fake exact cardinality with a monolithic predicate name.

### Scope with negation

Respect the surface scope.

- “Every X is not Y” → rule whose conclusion is `Not` of the Y proposition.
- “No X are Y” → usually same as universal negative: X implies not Y.
- “Some X are not Y” → existential witness with a `Not` fact.
- “Not every X is Y” → existential counterexample if one is asserted or known.
- “A student did not leave” is not the same as “No student left.”

`Not` scopes only over its nested expression.

---

## 5. Predicate atomicity — critical rule

Every predicate name must denote one reusable semantic primitive. Prefer 1–3 English words. The predicate should be able to appear as a premise, conclusion, or query target in other reasoning.

Do not encode the answer, a whole sentence, a comparison, a number, or a named-entity configuration into the predicate name.

Exactly four contrastive examples follow.

### 5.1 Numbers/cardinalities embedded in the name

BAD:

```metta
(: f_bad1 (TwoSurveyorsSigned report9) (STV 1.0 1.0))
```

GOOD:

```metta
(: f_good1 (Surveyor surveyorA) (STV 1.0 1.0))
(: f_good2 (Surveyor surveyorB) (STV 1.0 1.0))
(: f_good3 (Report report9) (STV 1.0 1.0))
(: f_good4 (Sign signEvt9) (STV 1.0 1.0))
(: f_good5 (Agent signEvt9 surveyorA) (STV 1.0 1.0))
(: f_good6 (Agent signEvt9 surveyorB) (STV 1.0 1.0))
(: f_good7 (Patient signEvt9 report9) (STV 1.0 1.0))
```

### 5.2 Thresholds or ranges embedded in the name

BAD:

```metta
(: r_bad2
   (Implication
      (Premises (PackageOver30Kg $p))
      (Conclusions (NeedsCart $p)))
   (STV 1.0 1.0))
```

GOOD:

```metta
(: r_good2
   (Implication
      (Premises
         (Package $p)
         (WeightKg $p $w)
         (GreaterThan $w 30))
      (Conclusions (NeedsCart $p)))
   (STV 1.0 1.0))
```

### 5.3 Multi-concept concatenation

BAD:

```metta
(: f_bad3 (CarefullySealedVialInLab rina vial7 lab2) (STV 1.0 1.0))
```

GOOD:

```metta
(: f_good31 (Seal sealEvt7) (STV 1.0 1.0))
(: f_good32 (Agent sealEvt7 rina) (STV 1.0 1.0))
(: f_good33 (Patient sealEvt7 vial7) (STV 1.0 1.0))
(: f_good34 (Manner sealEvt7 careful) (STV 1.0 1.0))
(: f_good35 (Location sealEvt7 lab2) (STV 1.0 1.0))
(: f_good36 (Vial vial7) (STV 1.0 1.0))
(: f_good37 (Lab lab2) (STV 1.0 1.0))
```

### 5.4 Entity-specific comparisons fused into the name

BAD:

```metta
(: f_bad4 (TowerATallerThanTowerB) (STV 1.0 1.0))
```

GOOD:

```metta
(: f_good41 (HeightM towerA 91) (STV 1.0 1.0))
(: f_good42 (HeightM towerB 84) (STV 1.0 1.0))
(: r_good4
   (Implication
      (Premises
         (HeightM $x $hx)
         (HeightM $y $hy)
         (GreaterThan $hx $hy))
      (Conclusions (TallerThan $x $y)))
   (STV 1.0 1.0))
```

---

## 6. Decomposition strategies

### Simple classification

Map “X is a Y” to a unary class predicate on the entity.

Use the class noun as the predicate. Do not create predicates like `IsADoctor`; use `Doctor`.

### Properties and attributes

For simple qualitative adjectives, use unary property predicates.

For measurable or gradable attributes, prefer value-bearing predicates with the value exposed as an argument. This allows comparisons and threshold rules.

Do not bake intensifiers or values into predicates. Use separate degree/value relations where needed.

### Events and thematic roles

For actions, especially transitive, ditransitive, modified, passive, or temporally located actions, use event reification:

- one event constant
- one event-type predicate from the verb lemma
- role predicates linking the event to participants

Use canonical roles such as:

- `Agent`
- `Patient`
- `Theme`
- `Recipient`
- `Beneficiary`
- `Instrument`
- `Source`
- `Destination`
- `Location`
- `Time`
- `Manner`

This keeps active/passive alternations and modifiers compositional.

### Modifiers

Do not fold modifiers into the main predicate.

Represent:

- adjectival modifiers as properties or value relations of the entity
- adverbs as `Manner` or another event-level modifier
- temporal modifiers as `Time`, `Before`, `After`, `During`, `StartsAt`, or `EndsAt`
- spatial modifiers as spatial relations or event path/location roles

### Relative clauses

Represent the head noun and the relative clause as separate facts or rule premises sharing the same entity variable/constant.

“The man who called” should not become one predicate. It is a `Man` entity plus a `Call` event whose agent is that entity.

### Coordination

For “and”, add multiple facts, multiple premises, or multiple conclusions as appropriate.

For coordinated noun phrases, share the same event/relation when both participants fill the same role.

For “or”, there is no documented disjunction operator. Use separate alternative facts/rules only when the source commits to each alternative; otherwise represent the unresolved alternative with a domain predicate such as `Alternative` or `Option`.

### Conditionals

Map “if/when/whenever/provided that” conditionals to rules. Put all conditions in premises and all consequences in conclusions.

Counterfactuals can use the same structural rule form only if the KB is meant to reason over hypothetical conditions; otherwise represent counterfactuality explicitly with a domain predicate.

---

## 7. Statements vs queries

Statements add facts or rules to the KB. Queries ask whether a proposition pattern is derivable.

Do not turn a generic English statement into a query for the rule itself unless the user is explicitly asking whether that rule is stored/provable as an object. Usually, add the rule as a statement and query a concrete conclusion.

For the question “Given that tagged samples are archived and sample7 is tagged, is sample7 archived?”:

BAD query:

```metta
(: $prf
   (Implication
      (Premises (Tagged $s))
      (Conclusions (Archived $s)))
   $tv)
```

GOOD: add the rule and fact, then query the target conclusion.

```metta
(: archiveTaggedRule
   (Implication
      (Premises (Tagged $s))
      (Conclusions (Archived $s)))
   (STV 1.0 1.0))
(: taggedSample7 (Tagged sample7) (STV 1.0 1.0))
(: $prf (Archived sample7) $tv)
```

Use a variable proof id in queries. Usually leave the truth-value position as `$tv`.

---

## 8. Query granularity

Statement decomposition and query decomposition are asymmetric.

Decompose rich statements into many atomic facts. But do not issue a parallel battery of queries that merely re-check every fact just added. Query only the unknown or conclusion requested by the English question.

Queries target one proposition pattern. Let the chainer search rules to prove that target. If a question requires multiple constraints, either:

- query stepwise, using returned bindings in later queries, or
- add/use a derived relation whose rule combines the constraints

For the question “What tool did Nora use in repair event repair17?” after the event has already been decomposed:

BAD:

```metta
(: $prf (Repair repair17) $tv)
(: $prf (Agent repair17 nora) $tv)
(: $prf (Patient repair17 pump9) $tv)
(: $prf (Instrument repair17 $tool) $tv)
```

GOOD:

```metta
(: $prf (Instrument repair17 $tool) $tv)
```

---

## 9. Built-in operators vs custom predicates

Use built-ins for structural reasoning:

- use `Implication` for conditionals, universals, generics, and rules
- use `Premises` and `Conclusions` to organize rule sides
- use `Not` for explicit negation
- use `GreaterThan` for numeric greater-than comparisons
- express less-than by reversing the arguments to `GreaterThan`
- use `Compute` for available computable functions
- use `FoldAll` / `FoldAllValue` for aggregation when the fold function is available
- use distribution operators for uncertain numeric values, not custom “uncertain value” predicates

Use custom predicates only for domain semantics: classes, event types, roles, attributes, relations, attitudes, discourse relations, and domain-specific measures.

Do not create custom versions of built-ins such as `IfThen`, `AndPremises`, `NotCat`, `MoreThanFive`, or `GreaterThanAliceBob`.

Do not use internal proof/runtime tokens as domain predicates.

---

## 10. Truth-value assignment

Use `STV 1.0 1.0` for direct, certain assertions from the source.

Use lower strength for uncertain truth claims when the English itself is epistemically uncertain, such as “probably”, “possibly”, or “it might be true that”, if the intended representation is graded belief in the proposition.

Do not use low STV as a substitute for logical negation. For “not P”, represent `Not` of P with appropriate confidence.

Do not use STV to represent measurement uncertainty. For uncertain numeric values, use the documented distribution truth/value forms and distribution operators.

For rules, the rule’s STV is the confidence in the rule itself, not the truth value of each future conclusion.

For modality:

- epistemic uncertainty may be represented with lower STV when appropriate
- ability, permission, obligation, and requirement should usually be represented as domain predicates such as `Able`, `Permitted`, `Required`, or `Obligated`, not as low-confidence occurrence of the embedded action

---

## 11. Phenomenon-specific mapping notes

### Determiners and definiteness

Indefinites introduce new witnesses unless already discourse-bound. Definites and demonstratives require coreference resolution to an existing constant. If unresolved, either create a context-specific constant only when the discourse clearly presupposes a specific entity, or leave the expression underspecified rather than pretending the identity is known.

### Possession, ownership, and association

Distinguish ownership, temporary possession, kinship/association, and part-whole relations when the wording supports it.

Use predicates such as `Owns`, `Possesses`, `AssociatedWith`, and `PartOf` rather than a generic overloaded `Has` when the distinction matters.

### Spatial relations and motion

Static spatial prepositions map to spatial relations such as `Under`, `Beside`, `On`, `Inside`, or `Near`.

Motion prepositions should attach to an event with roles such as `Source`, `Destination`, `Path`, or `Location`. Do not encode path into the motion predicate name.

### Temporal relations, tense, and aspect

Do not encode tense in the predicate. Represent time separately.

Use temporal relations such as `Before`, `After`, `During`, `Time`, `StartsAt`, and `EndsAt`.

Aspectual meanings should be separate properties of the event, such as `Ongoing`, `Completed`, or `Habitual`.

Future tense alone should not make the event a present fact unless the KB treats scheduled/planned events as facts; otherwise use a planning or scheduled-event predicate.

### Negation

Use `Not` for explicit denial. Avoid negative predicate names such as `NonMember`, `Unhappy`, or `DidNotCall` unless the English lexical item is genuinely atomic in the domain ontology.

Open-world caution: absence of a fact is not the same as `Not` of that fact.

### Modality

Represent ability, permission, obligation, necessity, and possibility explicitly when they are the object of reasoning.

Do not assert the embedded event as having occurred merely because someone can, may, must, or wants to do it.

### Comparisons

Expose the compared values as arguments and use `GreaterThan` where possible.

For “less than”, reverse the comparison.

For equality-like comparisons such as “as tall as”, use shared measured values if known, or a domain predicate such as `SameHeight` if equality is directly asserted and no equality primitive is available.

For superlatives, prefer a score/value plus comparison rules over predicates like `BestOption`; if “best” is merely asserted without data, `Best` may be used as a direct qualitative property.

### Propositional attitudes and embedded clauses

For belief, desire, hope, saying, knowing, and similar attitudes, do not automatically assert the embedded content as true.

Represent the attitude holder and content separately, using predicates such as `Believe`, `Want`, `Hope`, `Say`, `Know`, and `Content`.

For factive predicates such as “know”, only assert the embedded proposition separately if the KB policy treats the source as reliable and factivity is intended.

### Reported speech and quotation

Since there is no quoted-string syntax, direct quotes require an external symbol or reified content object.

Represent speech/writing as an event with `Agent`, `Recipient` if present, and `Content`.

### Passive voice and alternations

Passive voice changes surface subject, not semantic roles.

“The cake was eaten by the children” should use an eating event where the children are agents and the cake is patient/theme.

For agentless passives, leave the agent absent rather than inventing one.

For inchoatives such as “the door opened”, represent an opening event with the door as patient/theme and no agent unless one is stated.

### Transfer and ditransitives

Use event reification.

For giving, sending, telling, selling, lending, and similar predicates, represent:

- giver/sender/speaker as `Agent` or `Source`
- transferred item/message as `Theme` or `Patient`
- receiver as `Recipient`
- beneficiary separately as `Beneficiary` when distinct

### Causation and purpose

Use `Cause` to relate causing events/states to effects.

Use `Purpose` or `Goal` for intended outcomes. Do not assert the goal event occurred unless the English states that it did.

### Frequency and habituality

For repeated or habitual behavior, do not create many event facts unless individual occurrences are asserted.

Use `Habitual`, `Frequency`, `RecursOn`, or temporal schedule predicates as appropriate.

### Distributive and collective readings

For distributive “each”, attach the predicate to each member when members are known, or use a rule over `MemberOf`.

For collective “together”, represent one group event with a group agent and optionally mark it `Collective`.

Do not confuse “the students each lifted a box” with “the students together lifted the piano.”

### Relative clauses and nominal modification

Use shared variables or constants. Restrictive relative clauses add constraints; nonrestrictive relative clauses add additional facts about the same entity.

### Questions and wh-phrases

Map the wh-position to a variable in the queried proposition.

For event questions, usually query a role relation or a derived answer relation, not the entire decomposed event structure.

### Imperatives and directives

Commands, requests, permissions, and prohibitions should not be represented as completed actions.

Use directive predicates such as `Command`, `Request`, `Prohibit`, `Permitted`, or `Obligated`, with addressee and action content represented separately.

### Generic and kind-level statements

Generic kind statements usually become rules over instances.

Kind-level properties of the kind itself, such as “Tigers are endangered”, may be represented as a property of the class/kind if the intended subject is the species rather than each individual tiger.

### Measurement, amounts, and units

Represent the measured dimension and unit explicitly, preferably through a value predicate whose name fixes the unit, such as a height-in-centimeters or weight-in-kilograms relation.

Keep numeric values as arguments. Use distributions for uncertain measurements.

### Degree and intensification

Avoid predicates such as `VeryHot` or `ExtremelyHappy`.

Use measured values when possible. Otherwise use separate qualitative degree/intensity relations.

### Discourse sequencing and contrast

Temporal discourse markers such as “first”, “then”, and “afterward” map to temporal ordering among events.

Causal markers such as “so” and “therefore” map to `Cause` or to a rule only when the sentence states a general conditional.

Contrastive markers such as “however” can be represented with a discourse relation like `Contrast` only if discourse structure matters; otherwise they often do not affect factual content.

### Ellipsis and gapping

Recover omitted material from context before conversion. If the missing predicate or argument cannot be recovered, do not hallucinate a complete logical form.

### Clefts, focus, and “only”

Plain clefts usually preserve the same factual content while highlighting focus.

“Only” adds exclusivity. Because exclusivity requires reasoning over alternatives, represent it explicitly with an `Only` / `Exclusive` style predicate or generate negative facts for known alternatives when the domain is closed enough. Do not reduce “Only Maria laughed” to merely “Maria laughed.”