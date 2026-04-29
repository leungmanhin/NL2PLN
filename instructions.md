# NL to logic conversion instructions

You are converting English natural language (sentences or questions) into logical expressions (statements or queries) for a chainer/reasoner.  The chainer's syntax and semantics reference is provided separately as the `pln_spec` input (see `chainer_analysis.txt`).

## Chainer primitives
*Quick-reference cheat sheet of names with semantic backing.  Use these names verbatim wherever a primitive is needed.*

External API scaffolding

`(: proof-id proposition-or-rule truth-value)`
  Bare statement form for `add_atom`; `proof-id` must not be a variable and `truth-value` must be a supported non-variable TV/distribution form.

`(: $proofVar proposition-pattern $tvVar)`
  Bare query form for `query`; proof position must be a variable, and the TV position is normally a variable such as `$tv`.

`$variable`
  Variables begin with `$` and unify across a query/rule expression.

`constant`
  Any symbol not beginning with `$` is treated as a constant/predicate symbol and matches syntactically.

`(no_inverse proof-id)`
  Structured proof id wrapper that disables inverse implication support for that rule.

Do not emit Python API inputs wrapped as `!(compileadd ...)`, `!(query ...)`, or `!(eval ...)`; those are internal runtime commands, not external bare expressions.


Plain proposition shape

`(Predicate arg1 arg2 ...)`
  Ordinary user predicate/proposition form; predicate names have no built-in domain meaning but are matched and unified syntactically.


Rule structure

`(Implication (Premises premise1 premise2 ...) (Conclusions conclusion1 conclusion2 ...))`
  Rule form used inside a top-level statement.

`Premises`
  Required marker introducing rule antecedent expressions.

`Conclusions`
  Required marker introducing rule consequent expressions.


Computation and aggregation premises

`(Compute f (arg1 arg2 ...) -> $out)`
  Runs a reducible runtime function and binds the result to `$out`.

`(FoldAll pattern value init fold-fn -> out)`
  Folds over all facts matching `pattern`, extracting `value` from each match.

`(FoldAllValue pattern init fold-fn -> out)`
  Folds over truth values of all matches to `pattern`.

`(AverageDist pattern value -> out)`
  Aggregates matched distribution values into an average distribution.

`(MapDist f pattern value -> out)`
  Applies unary function `f` pointwise to a matched distribution.

`(Map2Dist f patternA valueA patternB valueB -> out)`
  Applies binary function `f` pointwise to two matched distributions.


Truth/proposition operators

`(Not expr)`
  Truth-value negation of an existing matched proof of `expr`; not negation-as-failure.

`(GreaterThan dist threshold)`
  Distribution-vs-numeric-threshold comparison, producing an STV.

`(> dist threshold)`
  Alias form for distribution-vs-threshold comparison.

`(GreaterThan distA distB)`
  Distribution-vs-distribution comparison when the second argument is a variable bound to a distribution.

`(> distA distB)`
  Alias form for distribution-vs-distribution comparison.

`(And expr1 expr2 ...)`
  Compound proposition whose child truth values are combined with `AndFormula`.

`(Or expr1 expr2 ...)`
  Compound proposition whose child truth values are combined with `OrFormula`.

`(LikelierThan expr1 expr2 ...)`
  Compound proposition comparing child proposition truth values by likelihood.


Supported statement truth/distribution forms

`(STV strength confidence)`
  Simple truth value with strength and confidence.

`(NatDist pairs)`
  Discrete natural-number distribution TV form accepted by the validator.

`(FloatDist pairs)`
  Discrete floating-point distribution TV form accepted by the validator.

`(ParticleDist ref)`
  Opaque particle distribution reference.

`(ParticleDist ref scale)`
  Opaque particle distribution reference with scale.

`(PointMass x)`
  Particle distribution constructor for an exact value.

`(ParticleFromNormal mu sigma)`
  Particle distribution constructor using a deterministic normal-like kernel.

`(ParticleFromPairs ((x1 w1) (x2 w2) ...))`
  Particle distribution constructor from weighted samples.


Built-in reducer/formula names

`Equal`
  Syntactic equality reducer, typically used as `(Compute Equal (arg1 arg2) -> $out)`.

`NotEqual`
  Syntactic inequality reducer, typically used as `(Compute NotEqual (arg1 arg2) -> $out)`.

`MpFormula`
  Internal implication modus-ponens truth-value combination formula.

`AndFormula`
  Internal TV formula backing `And`.

`OrFormula`
  Internal TV formula backing `Or`.

`LikelierThanFormula`
  Internal TV formula backing `LikelierThan`.

`AndProjection`
  Internal projection support for compound `And` sources.

`OrProjection`
  Internal projection support for compound `Or` sources.

`DistGreaterThanFormula`
  Internal distribution-vs-threshold comparison formula backing `GreaterThan`.

`DistGreaterThanDistFormula`
  Internal distribution-vs-distribution comparison formula backing `GreaterThan`.

`NotFormula`
  Internal TV formula backing `Not`.

`InversionFormula`
  Internal formula used for inverse implication support.

`ParticleFromPairs`
  Distribution constructor/function for weighted particles.

`PointMass`
  Distribution constructor/function for exact point values.

`ParticleFromNormal`
  Distribution constructor/function for normal-like particle samples.

`ParticlePairs`
  Advanced particle distribution utility.

`ParticleMap`
  Advanced unary particle-map utility.

`ParticleMap2`
  Advanced binary particle-map utility.

`ParticleAddBernoulliFromSTV`
  Advanced fold helper for adding Bernoulli contributions from STV values.

`DistMapFormula`
  Internal formula backing `MapDist`.

`DistMap2Formula`
  Internal formula backing `Map2Dist`.

`DistSumCountAcc`
  Internal accumulator helper for distribution averaging.

`DistAverageFromSumCount`
  Internal finish helper for distribution averaging.

`ParticleStoreCount`
  Advanced particle-store utility.

`ParticleStoreClear`
  Advanced particle-store utility.

`ParticleStorePruneKB`
  Advanced particle-store utility.

`ParticleSetBudget`
  Advanced particle-budget utility.

`ParticleGetBudget`
  Advanced particle-budget utility.

---

## Suggested vocabulary
*Naming suggestions for non-primitive concepts (thematic roles, spatial/temporal relations, etc.).  Use for consistency across translations; these names do NOT have chainer semantics unless explicit rules are added to the KB.*

IMPORTANT: The names below are NAMING SUGGESTIONS ONLY for ordinary predicates and domain concepts. They are not PeTTaChainer primitives and have no built-in semantics unless a name is separately listed under `chainer_primitives`. Use them for consistency when creating user-level facts, rules, event roles, and query predicates.

Open-class predicate template families

Entity classes and kinds
  `(Kind entity)`
  Use UpperCamelCase noun lemmas as unary predicates: `Person`, `Dog`, `City`, `Room`, `Organization`, `Event`, `Object`, `Substance`.

Event types
  `(EventType event)`
  Use UpperCamelCase verb/event lemmas as unary predicates: `Break`, `Open`, `Give`, `Arrive`, `Build`, `Destroy`, `Say`, `Believe`.

Boolean properties and states
  `(Property entity)` `(State entity)`
  Use reusable adjective/state names: `Open`, `Closed`, `Broken`, `Available`, `Empty`, `Ripe`, `Valid`, `Finished`.

Attribute values
  `(Attribute entity value)`
  Canonical families: `Color`, `Shape`, `Material`, `Size`, `Status`, `Role`, `EyeColor`, `Language`, `NameForm`.

Measurements and distributions
  `(DimensionDist entity dist)`
  Canonical families: `HeightDist`, `LengthDist`, `WidthDist`, `DepthDist`, `WeightDist`, `MassDist`, `TemperatureDist`, `SpeedDist`, `DistanceDist`, `AreaDist`, `VolumeDist`, `AgeDist`, `CountDist`, `ScoreDist`.

Aggregates and derived quantities
  `(AvgDimensionDist group dist)` `(TotalDimension collection value)` `(Count collection n)`
  Canonical families: `AvgHeightDist`, `AvgWeightDist`, `TotalScore`, `TotalCost`, `Count`, `MemberCount`, `RemainingCount`.

Domain-specific relations
  `(Relation arg1 arg2 ...)`
  Use concise UpperCamelCase lemmas for stable reusable relations: `Own`, `Employs`, `Teaches`, `LivesIn`, `DependsOn`, `Represents`, `Requires`.


Reference, names, and identity
  `(Named entity name)` `(AliasOf entity name)` `(SurfaceForm entity form)` `(SameEntity entity1 entity2)` `(Distinct entity1 entity2)` `(RefersTo mention entity)` `(Corefers mention1 mention2)`
  Disambiguation notes:
  - SameEntity vs AliasOf: `SameEntity` relates two entity constants; `AliasOf` or `Named` relates an entity to a name/form.
  - Distinct records domain-level difference; it does not perform syntactic inequality by itself.

Kind-level and taxonomy relations
  `(SubkindOf kind superkind)` `(KindProperty kind property)` `(GenericOf statement kind)` `(FormerKind entity kind)` `(FakeKind entity kind)` `(AllegedKind source entity kind)`
  Disambiguation notes:
  - For ordinary instance membership, prefer unary class predicates such as `(Dog fido)`; use these names for explicit meta-level kind relations.
  - FakeKind and AllegedKind should not imply actual membership in the named kind unless additional rules say so.

Possession, containment, part-whole, and membership
  `(Own owner item)` `(BelongsTo item owner)` `(Possess possessor item)` `(Controls controller item)` `(Contains container content)` `(PartOf part whole)` `(ComponentOf component whole)` `(PortionOf portion whole)` `(MemberOf member group)` `(InSet item set)` `(IncludedIn item set)` `(ExcludedFrom item set)` `(AssociatedWith entity1 entity2)`
  Disambiguation notes:
  - Own vs Possess: `Own` is legal/social ownership; `Possess` is current having/control.
  - Contains vs PartOf: `Contains` is containment; `PartOf` is structural/constitutive relation.
  - MemberOf vs PartOf: `MemberOf` is group/set membership; `PartOf` is part-whole structure.

Thematic and event roles
  `(Agent event entity)` `(Patient event entity)` `(Theme event entity)` `(Experiencer event entity)` `(Stimulus event entity)` `(Recipient event entity)` `(Beneficiary event entity)` `(Source event entity)` `(Goal event entity)` `(Instrument event entity)` `(Location event place)` `(Path event path)` `(Manner event manner)` `(ResultState event state)` `(InitialState event state)` `(FinalState event state)` `(CreatedEntity event entity)` `(DestroyedEntity event entity)` `(Content event proposition)` `(Topic event entity)` `(Speaker event entity)` `(Addressee event entity)`
  Disambiguation notes:
  - Patient vs Theme: `Patient` is affected or changed; `Theme` is moved, transferred, perceived, or central without necessarily changing.
  - Agent vs Experiencer: `Agent` intentionally acts; `Experiencer` perceives, feels, or cognizes.
  - Recipient vs Beneficiary: `Recipient` receives a theme; `Beneficiary` benefits from the event.
  - Source vs Goal: `Source` is origin; `Goal` is destination or endpoint.

Spatial relations
  `(In entity container)` `(Inside entity container)` `(Outside entity landmark)` `(On entity support)` `(At entity place)` `(Under entity landmark)` `(Over entity landmark)` `(Above entity landmark)` `(Below entity landmark)` `(NextTo entity landmark)` `(AdjacentTo entity landmark)` `(Near entity landmark)` `(FarFrom entity landmark)` `(Between entity landmark1 landmark2)` `(Around entity landmark)` `(AcrossFrom entity landmark)` `(NorthOf entity landmark)` `(SouthOf entity landmark)` `(EastOf entity landmark)` `(WestOf entity landmark)` `(LeftOf entity landmark)` `(RightOf entity landmark)` `(InFrontOf entity landmark)` `(Behind entity landmark)` `(Along entity path)` `(Through entity path)` `(Toward entity goal)` `(AwayFrom entity source)`
  Disambiguation notes:
  - In vs Inside: `In` is general containment/location; `Inside` emphasizes interior containment.
  - On vs At: `On` implies support/contact; `At` is general location.
  - Above vs Over: `Above` is vertical ordering; `Over` may imply covering or traversal.
  - LeftOf/RightOf require an explicit or understood viewpoint.

Deictic and context-relative reference
  `(ContextSpeaker context entity)` `(ContextAddressee context entity)` `(Viewpoint context entity)` `(DeicticAnchor context entity)` `(Here place context)` `(There place context)` `(Now time context)` `(Today date context)` `(Yesterday date context)` `(Tomorrow date context)` `(Proximal entity context)` `(Distal entity context)`
  Disambiguation notes:
  - ContextSpeaker vs Speaker: `ContextSpeaker` identifies the utterance speaker; `Speaker` is an event role for a communication event.
  - Proximal vs Distal: `Proximal` corresponds to “this/here”; `Distal` corresponds to “that/there.”

Temporal relations
  `(Time eventuality time)` `(StartTime eventuality time)` `(EndTime eventuality time)` `(Duration eventuality duration)` `(Before eventuality1 eventuality2)` `(After eventuality1 eventuality2)` `(During eventuality interval)` `(Within eventuality interval)` `(Overlaps eventuality1 eventuality2)` `(Simultaneous eventuality1 eventuality2)` `(StartsAt interval time)` `(EndsAt interval time)` `(ContainsInterval interval subinterval)` `(ReferenceTime context time)` `(UtteranceTime context time)`
  Disambiguation notes:
  - Time vs StartTime: `Time` is a general temporal location; `StartTime` marks onset.
  - During vs Overlaps: `During` implies containment in an interval; `Overlaps` only requires temporal intersection.
  - Before vs After: use one consistently rather than asserting both unless needed.

Aspect, recurrence, and frequency
  `(Ongoing eventuality)` `(Completed eventuality)` `(Incomplete eventuality)` `(Habitual eventuality)` `(Repeated eventuality)` `(RepetitionCount eventuality n)` `(Frequency eventuality frequency)` `(Usually proposition)` `(Still eventuality)` `(Already eventuality)` `(Again eventuality)` `(Stopped eventuality)` `(Resumed eventuality)`
  Disambiguation notes:
  - Habitual vs Repeated: `Habitual` is a general pattern; `Repeated` concerns multiple occurrences.
  - Completed vs Already: `Completed` is event status; `Already` is aspect relative to expectation/reference time.

Causation, purpose, reason, and means
  `(Causes cause effect)` `(CausedBy effect cause)` `(Enables condition event)` `(Prevents preventer event)` `(ReasonFor event reason)` `(PurposeOf event goal)` `(IntendedResult event result)` `(Means event method)` `(Method event method)` `(ByMeansOf event means)`
  Disambiguation notes:
  - Cause vs ReasonFor: `Causes` is causal production; `ReasonFor` records explanation/motivation.
  - PurposeOf vs IntendedResult: `PurposeOf` is the goal of acting; `IntendedResult` is the desired outcome state.
  - Instrument vs Means: use `Instrument` for a concrete tool role; use `Means` or `Method` for a procedure.

Change of state, creation, destruction, and existence
  `(Becomes entity state)` `(ChangesFrom entity oldState)` `(ChangesTo entity newState)` `(Creates event entity)` `(Destroys event entity)` `(Exists entity)` `(ExistedAt entity time)` `(CeasesToExist entity time)` `(Absent entity context)` `(Missing entity context)`
  Disambiguation notes:
  - Absent vs Missing: `Absent` is not present in a context; `Missing` implies expected presence.
  - Destroys vs CeasesToExist: `Destroys` is an event relation; `CeasesToExist` is an existence/status relation.

Modality, ability, permission, and norms
  `(Able actor action)` `(CapableOf entity action)` `(Possible proposition)` `(Probable proposition)` `(Necessary proposition)` `(Obligated actor action)` `(Required action)` `(Should actor action)` `(Permitted actor action)` `(Forbidden actor action)` `(AllowedBy authority actor action)` `(ProhibitedBy authority actor action)` `(Requested requester addressee action)` `(Commanded commander addressee action)`
  Disambiguation notes:
  - Possible vs Probable: `Possible` marks compatibility; `Probable` marks likelihood.
  - Necessary vs Obligated: `Necessary` is proposition-level necessity; `Obligated` is an agent norm.
  - Forbidden vs ProhibitedBy: `Forbidden` records the norm; `ProhibitedBy` records its authority/source.

Propositional attitudes, desires, and plans
  `(Believes agent proposition)` `(Knows agent proposition)` `(Suspects agent proposition)` `(Doubts agent proposition)` `(Wants agent content)` `(Hopes agent proposition)` `(Intends agent action)` `(Plans agent action)` `(Fears experiencer content)` `(Realizes agent proposition)` `(Regrets agent proposition)` `(Remembers agent proposition)` `(Forgets agent proposition)`
  Disambiguation notes:
  - Believes vs Knows: `Knows` is factive in ordinary English; `Believes` is not.
  - Wants vs Intends: `Wants` is desire; `Intends` commits the agent toward action.
  - Hopes vs Wants: `Hopes` usually embeds an uncertain proposition; `Wants` can target an action or state.

Communication and reported content
  `(Says speaker content)` `(States speaker content)` `(Claims speaker content)` `(Reports reporter content)` `(Tells speaker addressee content)` `(Asks speaker addressee content)` `(Answers speaker addressee content)` `(Promises speaker addressee content)` `(Informs speaker addressee content)` `(ReportedBy content source)`
  Disambiguation notes:
  - Says vs Claims: `Claims` suggests a commitment that may be disputed; `Says` is neutral speech reporting.
  - Tells vs Informs: `Informs` suggests successful communication; `Tells` only records the act.

Evidentiality and information source
  `(ObservedBy proposition observer)` `(HeardBy proposition hearer)` `(ReportedBy proposition source)` `(InferredFrom proposition evidence)` `(EvidenceFor evidence proposition)` `(EvidenceAgainst evidence proposition)` `(Apparently proposition)` `(Seems proposition)` `(SourceConfidence source confidence)` `(Reliability source reliability)`
  Disambiguation notes:
  - Apparently vs Seems: `Apparently` is report/inference-like; `Seems` often marks appearance or subjective evidence.
  - SourceConfidence vs Reliability: confidence may be claim-specific; reliability is source-level.

Negation, absence, exclusion, and incompatibility
  `(Denied proposition source)` `(FalseInContext proposition context)` `(Absent entity context)` `(Missing entity context)` `(Unavailable entity context)` `(NotMemberOf entity group)` `(ExcludedFrom entity set)` `(CounterexampleTo entity claim)` `(Contradicts proposition1 proposition2)` `(IncompatibleWith proposition1 proposition2)`
  Disambiguation notes:
  - Denied vs FalseInContext: `Denied` records a denial act/source; `FalseInContext` records modeled falsity.
  - NotMemberOf vs ExcludedFrom: `NotMemberOf` is membership negation; `ExcludedFrom` implies active exclusion or rule-based exclusion.

Quantification, cardinality, proportions, and exceptions
  `(Cardinality collection n)` `(MinCardinality collection n)` `(MaxCardinality collection n)` `(ApproxCardinality collection n)` `(Proportion collection property ratio)` `(MostOf collection property)` `(ManyOf collection property)` `(FewOf collection property)` `(SeveralOf collection property)` `(HalfOf collection property)` `(AllExcept collection exception)` `(ExceptionTo exception ruleOrSet)` `(OnlySatisfier entity condition)`
  Disambiguation notes:
  - Cardinality vs Count: use `Cardinality` for asserted set size; use `Count`-style predicates for derived numeric quantities.
  - MostOf/FewOf/ManyOf are vague/proportional labels unless rules define thresholds.
  - ExceptionTo relates an exception to a rule or set; it does not automatically retract the general rule.

Comparison, degree, ranking, and sequence
  `(DegreeOf entity dimension valueOrDist)` `(GreaterDegreeThan entity1 entity2 dimension)` `(LessDegreeThan entity1 entity2 dimension)` `(EqualDegree entity1 entity2 dimension)` `(RankInSet entity set rank dimension)` `(BestInSet entity set criterion)` `(WorstInSet entity set criterion)` `(FirstInSequence entity sequence)` `(LastInSequence entity sequence)` `(NextInSequence entity1 entity2 sequence)` `(PreviousInSequence entity1 entity2 sequence)` `(OrdinalInSequence entity sequence ordinal)`
  Disambiguation notes:
  - Degree comparison names are ordinary predicates; use distribution/comparison primitives separately when numeric reasoning is required.
  - RankInSet gives an explicit rank; BestInSet/WorstInSet identify extrema by a criterion.

Collections, plurality, mass nouns, and distributivity
  `(Group group)` `(Collection collection)` `(Set set)` `(MemberOf member group)` `(SubgroupOf subgroup group)` `(Substance substance)` `(QuantityOf quantity substance)` `(PortionOf portion substance)` `(UnitOf quantity unit)` `(CollectiveParticipant event group)` `(DistributiveOver event group)` `(EachMember member group)`
  Disambiguation notes:
  - Group vs Collection: `Group` can act collectively; `Collection` is any aggregate.
  - Substance vs PortionOf: `Substance` names mass material; `PortionOf` identifies a bounded amount.

Clause embedding, modification, and attachment
  `(EmbeddedContent host proposition)` `(ComplementOf host proposition)` `(RestrictiveModifier entity condition)` `(NonrestrictiveInfo entity proposition)` `(Appositive entity description)` `(ModifierOf modifier target)` `(AttachmentReading analysis target)` `(ControlSubject embeddedEvent controller)` `(RaisedSubject embeddedPredicate entity)` `(EllipsisResolution fragment proposition)`
  Disambiguation notes:
  - RestrictiveModifier vs NonrestrictiveInfo: restrictive information narrows reference; nonrestrictive information adds side information.
  - ControlSubject vs RaisedSubject: control supplies an understood actor; raising marks the subject of the embedded predicate.

Focus, discourse status, and presupposition
  `(FocusOf proposition focus)` `(OnlyFocus proposition focus)` `(AlsoFocus proposition focus)` `(EvenFocus proposition focus)` `(TopicOf discourse entity)` `(Backgrounded proposition)` `(Presupposes trigger proposition)` `(PresupposedIn proposition context)` `(ContrastSet focus set)`
  Disambiguation notes:
  - OnlyFocus restricts alternatives; AlsoFocus adds another true alternative; EvenFocus marks unexpectedness.
  - Presupposes records background commitment; it does not itself decide whether to assert the presupposed proposition.

Questions, directives, and speech acts
  `(Question act)` `(YesNoQuestion act proposition)` `(WhQuestion act variable)` `(Answer answer question)` `(Directive act)` `(Imperative act)` `(Request speaker addressee action)` `(Command speaker addressee action)` `(Instruction speaker addressee action)` `(PermissionGrant authority actor action)` `(Prohibition authority actor action)`
  Disambiguation notes:
  - Question predicates are for storing/reporting question acts; actual PeTTaChainer queries use the query scaffold.
  - Request vs Command: `Request` is weaker/socially optional; `Command` implies authority.

Non-intersective and intensional modifiers
  `(Former entity roleOrKind)` `(Fake entity kind)` `(Alleged source entity kindOrProposition)` `(Potential entity kind)` `(PossibleKind entity kind)` `(Prospective entity kind)` `(Counterfeit entity kind)` `(Nominal entity kind)`
  Disambiguation notes:
  - Former does not imply current role/kind membership.
  - Fake, Alleged, Potential, and PossibleKind do not imply actual membership in the modified kind.

---

## Conversion guidelines
*Mapping rules from English to the chainer's logic.  Apply these patterns; cross-reference the vocabulary above for canonical names.*

# Conversion Guidelines for English → PeTTaChainer Logic

## 1. Overall conversion workflow

1. Decide the speech act:
   - Declarative fact → add one or more factual atoms.
   - Generic, universal, conditional, or causal regularity → add an implication rule.
   - Question → issue a query only; do not add the queried proposition as a fact.
   - Directive/imperative → usually represent as a requested/obligated action, not as an action that already happened.

2. Resolve references before emitting logic:
   - Map each entity mention to a stable constant.
   - Resolve pronouns, definites, demonstratives, and ellipsis from context when possible.
   - If unresolved, create a context-specific placeholder constant rather than a global ambiguous symbol.

3. Decompose English into atomic predicates linked by shared constants or variables.

4. Assign an appropriate truth value:
   - Use strong truth for direct reliable assertions.
   - Use lower strength/confidence for uncertain, reported, modal, vague, or defeasible information.
   - Use distribution values for uncertain numeric measurements, not STV strength.

5. For questions, query only the unknown target requested by the English question.

---

## 2. Naming conventions

### Predicates

Use UpperCamelCase predicate names derived from English lemmas.

Preferred patterns:

- Class/type nouns: `Dog`, `Person`, `Room`, `Bridge`.
- Simple properties/states: `Open`, `Broken`, `Available`.
- Attribute dimensions: `Color`, `Shape`, `TemperatureDist`, `LengthDist`.
- Relations: `Own`, `PartOf`, `MemberOf`, `LocatedIn`, `NextTo`.
- Event types: `Break`, `Give`, `Arrive`, `Open`.
- Event-role predicates: `Agent`, `Patient`, `Theme`, `Recipient`, `Source`, `Goal`, `Instrument`, `Location`, `Time`, `Duration`, `Manner`, `ResultState`.

Use lemmatized forms:
- “dogs” → `Dog`
- “broke/breaks/breaking” → `Break`
- “children” → `Child`
- “is taller than” → `TallerThan` or a rule over `HeightDist` plus `GreaterThan`

Avoid encoding tense, determiners, auxiliaries, quantifiers, entity names, numbers, thresholds, or whole clauses in predicate names.

### Constants

Use stable, unique constants for entities.

Recommended style:
- `person_alice_001`
- `city_paris_fr`
- `city_paris_tx`
- `dog_fido_001`
- `event_give_042`

In small closed examples, simple constants like `alice`, `fido`, or `room1` are acceptable if there is no ambiguity.

Surface names should not be treated as guaranteed unique real-world identities. Prefer a unique entity id plus a naming fact using a custom predicate such as `Named`.

### Variables

Use meaningful lowercase variable names beginning with `$`:
- `$person`
- `$dog`
- `$event`
- `$room`
- `$heightDist`
- `$tv`
- `$prf`

Use the same variable only when the English requires coreference. Use different variables for distinct participants unless equality/coreference is intended.

### Proof ids

Use unique, readable proof ids for facts and rules. For one-way English conditionals where inverse inference would be misleading, prefer a proof id wrapped with `no_inverse`.

---

## 3. Entity representation and identity

Constants are syntactic labels. The chainer does not perform entity resolution.

### Named entities

For robust modeling:

- Create a unique entity constant.
- Add type/class facts when known.
- Add a naming relation if the surface name matters.

Do not rely on the surface name alone when two entities can share it.

### Same surface name, different entities

Disambiguate by type, location, context, or index:
- Paris, France → `city_paris_fr`
- Paris, Texas → `city_paris_tx`
- two people named Ben → `person_ben_001`, `person_ben_002`

### Identity statements

For English identity such as “Clark Kent is Superman,” the preferred representation is to use one entity id and attach both names or aliases to it. If two constants have already been introduced, a custom predicate such as `SameEntity` or `AliasOf` may record the assertion, but it will not automatically make the constants interchangeable unless you add rules for that behavior.

### Difference statements

For “Tom is not Jerry,” use distinct constants. If the distinction itself must be queryable, add a custom relation such as `Distinct`. For rule premises requiring syntactic inequality, use `Compute NotEqual`.

### First- and second-person pronouns

Do not use bare global constants like `I`, `we`, or `you`.

If known:
- Map “I” to the speaker’s entity id.
- Map “you” to the addressee’s entity id.
- Map “we” to a group entity or to explicit members.

If unknown:
- Use context-specific placeholders such as `unknown_speaker_ctx17` or `unknown_addressee_ctx17`.

---

## 4. Quantification and scope

### Universal quantification

Map “all,” “every,” “each,” generic plurals, and ordinary conditionals to implication rules with variables.

The restrictor becomes one or more premises. The scope/body becomes the conclusion. Variables in a rule are implicitly universal over rule matches.

Examples of English patterns:
- “Every dog is an animal” → premise `Dog` over `$x`, conclusion `Animal` over `$x`.
- “Every student who passed received a certificate” → premises for `Student` and `Passed`, conclusion for the receiving/certificate relation.
- “If the alarm rings, leave the building” → premise for the alarm event/state, conclusion for the required leaving action or obligation.

Use `no_inverse` for rules that should not support reverse/inverse inference.

### Existential quantification

There is no source-level existential operator.

For assertions like “A dog barked” or “Someone called,” introduce a witness constant scoped to the discourse/context:
- create an entity/event id;
- assert its type;
- assert the described relation/event.

For questions like “Who called?” use a query variable.

Do not assert ordinary facts with free variables to mean “there exists”; use witness constants.

### Existentials inside universals

For “Every child received a sticker,” the English reading is universal-existential. If concrete instances are known, create actual sticker witnesses for each child. If the statement is a general entitlement or obligation rather than observed individual stickers, represent that directly with predicates such as `EntitledTo`, `Assigned`, `Obligated`, or a domain-specific relation. Use conclusion-only variables only with care, because existential/skolem-like behavior is internal and can be opaque.

### Cardinality

Do not encode cardinality in predicate names.

For small explicit assertions, introduce witness constants and, when needed, a `Cardinality` fact or `Distinct` facts.

For derived counts, use aggregation rules with `FoldAll`, `FoldAllValue`, or distribution helpers, then query the resulting count/count-distribution predicate. For exact numeric checks in rules, use `Compute Equal` where appropriate. For probability-style threshold checks over distributions, expose the distribution and use `GreaterThan`.

### Negation and quantifier scope

There is no general syntax for nested first-order quantifier scope. Preserve intended readings by making witnesses, restrictors, exceptions, counts, and negative evidence explicit.

Important distinctions:
- “Not every X is Y” requires a counterexample or count/proportion model.
- “No X is Y” requires explicit negative evidence, a count of zero, or a domain-specific absence/exclusion predicate.
- Absence of proof is not proof of absence.

---

## 5. Predicate atomicity

Every predicate name must denote one semantic primitive. A predicate should be reusable in other rules and queries. If the predicate name describes the entire answer, the reasoning has been hidden inside the string and the chainer cannot compose with it.

### 1. Numbers/cardinalities embedded in the predicate name

BAD:
```metta
(: bad1 (OwnsTwoBicycles rina) (STV 1.0 1.0))
```

GOOD:
```metta
(: bike1 (Bicycle bicycle_rina_001) (STV 1.0 1.0))
(: bike2 (Bicycle bicycle_rina_002) (STV 1.0 1.0))
(: own1 (Own rina bicycle_rina_001) (STV 1.0 1.0))
(: own2 (Own rina bicycle_rina_002) (STV 1.0 1.0))
(: distinct1 (Distinct bicycle_rina_001 bicycle_rina_002) (STV 1.0 1.0))
(: card1 (Cardinality (BicyclesOwnedBy rina) 2) (STV 1.0 1.0))
```

### 2. Numerical thresholds/ranges embedded in the predicate name

BAD:
```metta
(: bad2
   (Implication
      (Premises (GreenhouseAbove18Degrees $g))
      (Conclusions (Ventilate $g)))
   (STV 1.0 1.0))
```

GOOD:
```metta
(: greenhouseTemp1 (TemperatureDist greenhouse7 (PointMass 19.5)) (STV 1.0 1.0))

(: good2
   (Implication
      (Premises
         (Greenhouse $g)
         (TemperatureDist $g $tempDist)
         (GreaterThan $tempDist 18.0))
      (Conclusions (Ventilate $g)))
   (STV 1.0 1.0))
```

### 3. Multi-concept concatenation

BAD:
```metta
(: bad3 (QuickKitchenWindowOpening noah window9) (STV 1.0 1.0))
```

GOOD:
```metta
(: openEvt1 (Open open_evt_001) (STV 1.0 1.0))
(: openAgent1 (Agent open_evt_001 noah) (STV 1.0 1.0))
(: openPatient1 (Patient open_evt_001 window9) (STV 1.0 1.0))
(: openManner1 (Manner open_evt_001 quickly) (STV 1.0 1.0))
(: window1 (Window window9) (STV 1.0 1.0))
(: windowLoc1 (Location window9 kitchen1) (STV 1.0 1.0))
(: kitchen1Fact (Kitchen kitchen1) (STV 1.0 1.0))
```

### 4. Entity-specific comparisons fused into the predicate name

BAD:
```metta
(: bad4 (LochNessDeeperThanLakeTahoe) (STV 1.0 1.0))
```

GOOD:
```metta
(: depth1 (DepthDist loch_ness (PointMass 227.0)) (STV 1.0 1.0))
(: depth2 (DepthDist lake_tahoe (PointMass 501.0)) (STV 1.0 1.0))

(: good4
   (Implication
      (Premises
         (DepthDist $placeA $depthA)
         (DepthDist $placeB $depthB)
         (GreaterThan $depthA $depthB))
      (Conclusions (DeeperThan $placeA $placeB)))
   (STV 1.0 1.0))
```

---

## 6. Decomposition strategies

### Decompose complex noun phrases

For “the red cube on the table,” do not create one predicate for the whole phrase. Represent:
- entity type: `Cube`
- property: `Color` or `Red`
- spatial relation: `On`

Use the same entity constant or variable across these atoms.

### Decompose events with roles

Use event reification when an action has:
- more than two participants,
- time/location/manner/instrument modifiers,
- passive voice,
- causation/purpose,
- result state,
- repetition,
- uncertainty,
- or needs to be referred to later.

Represent the event type separately from role predicates. For simple stable binary relations with no event modifiers, a direct binary predicate is acceptable.

### Coordinate structures

For “A and B”:
- If English asserts two independent facts, add two facts.
- If a query asks for something satisfying multiple conditions, use a compound query with `And`.
- In rule antecedents, prefer multiple premises rather than hiding a conjunction inside a predicate name.

For “A or B”:
- Use `Or` only when the proposition itself is genuinely disjunctive.
- If the alternatives are separate possibilities with different evidence, represent them as separate uncertain facts.
- Exclusive-or requires explicit domain modeling, such as a custom incompatibility relation or cardinality constraint.

### Relative clauses

Restrictive relative clauses add conditions sharing the same variable/entity.

Nonrestrictive clauses and appositives usually add separate facts about an already identified entity.

### Modifier attachment ambiguity

Do not assert multiple readings unless the English/context supports them. Choose one attachment and encode that reading explicitly through shared variables/constants and role predicates.

---

## 7. Built-ins vs. custom predicates

Use built-ins when the English requires logical, numeric, distributional, or proof-compositional behavior:

- Use `Implication` for conditionals, generics, universal rules, and causal/defeasible rules.
- Use multiple premises for conjunctive conditions.
- Use `And` and `Or` for compound propositions/queries when the compound itself is the target.
- Use `Not` only for truth-value negation of an already matched proposition; never use it as negation-as-failure.
- Use `GreaterThan` for distribution-vs-threshold or distribution-vs-distribution comparisons.
- Use `Compute` for syntactic equality/inequality checks and runtime-computable arithmetic inside rules.
- Use `FoldAll`, `FoldAllValue`, `AverageDist`, `MapDist`, and `Map2Dist` for aggregation and distribution transformation.
- Use `LikelierThan` for likelihood/probability comparison of proposition truth values, not for ordinary degree comparison like height or weight.

Use custom predicates for domain semantics:
- `Own`, `PartOf`, `MemberOf`, `Color`, `Open`, `Believes`, `Obligated`, `Causes`, `PurposeOf`, etc.

Do not create custom predicates that duplicate built-in comparison, aggregation, or distribution operations.

---

## 8. Query semantics

Statements and queries are different speech acts.

- A statement adds a fact or rule to the KB.
- A query asks the chainer to prove a proposition pattern.
- Query proof ids must be variables.
- Query the proposition you want derived, not the rule surface form that might derive it.

### Rule-vs-query separation

If the KB contains a generic rule and facts, ask for the derived fact.

BAD query:
```metta
(: $prf
   (Implication
      (Premises (Falcon skye))
      (Conclusions (Bird skye)))
   $tv)
```

GOOD query:
```metta
(: $prf (Bird skye) $tv)
```

Use implication-shaped queries only for advanced/internal cases where the intended target really is an implication expression itself.

---

## 9. Query granularity

Statement decomposition and query decomposition are asymmetric.

For assertions, decompose into reusable atomic facts. For questions, do not issue a battery of queries that merely re-check each decomposed fact. Query the specific unknown requested by the question.

Use:
- concrete constants for yes/no questions;
- variables for wh-questions;
- compound `And` queries when the question asks for entities satisfying multiple conditions;
- derived aggregate predicates for “how many,” “average,” “total,” or ranking questions.

BAD query set for “Which blue cup is on shelf3?”:
```metta
(: $prf (Cup $x) $tv)
(: $prf (Blue $x) $tv)
(: $prf (On $x shelf3) $tv)
```

GOOD query:
```metta
(: $prf (And (Cup $x) (Blue $x) (On $x shelf3)) $tv)
```

---

## 10. Truth value assignment strategies

### Ordinary factual assertions

Use high strength and high confidence for direct, reliable assertions.

### Uncertain statements

For “probably,” “apparently,” “may,” “might,” “seems,” hearsay, or weak evidence:
- reduce confidence when the source is unreliable or indirect;
- reduce strength when the content itself is unlikely or probabilistic;
- optionally add source/evidence predicates such as `ReportedBy`, `ObservedBy`, or `EvidenceSource`.

### Generic/default rules

Strict definitional rules can use high truth values. Defeasible generics such as “birds fly” should use lower strength and/or confidence.

### Negated assertions

For direct denial of a positive predicate, represent the positive predicate with low strength and appropriate confidence, or use a lexical antonym/state predicate when the antonym is semantically primitive.

Do not use missing facts as negative evidence.

### Measurements

Do not encode numeric measurements in STV strength.

Use distribution-valued measurement predicates for:
- exact values;
- approximate values;
- noisy measurements;
- derived numeric quantities.

When using distribution helper premises later, put the distribution value as an argument of the proposition.

Normalize units externally when possible. If units must remain explicit, model the unit as a value/argument or use a carefully named dimension predicate, but never bake the numeric value into the predicate name.

---

## 11. Phenomenon-specific mapping notes

Only non-default or pitfall-prone phenomena are listed here. For ordinary classification, simple properties, simple relations, conjunction, and direct questions, apply the general rules above.

### Properties, attributes, and measurements

Use unary predicates for simple boolean properties and state predicates. Use dimension predicates such as `Color`, `Shape`, `Material`, `TemperatureDist`, `LengthDist`, `WeightDist`, or `HeightDist` when values need to be queried, compared, aggregated, or transformed.

Gradable adjectives such as “warm,” “tall,” “heavy,” and “near” should usually be derived from a measured distribution plus a threshold/context rule, rather than encoded as one-off threshold predicates.

### Possession, containment, part-whole, and membership

English “has” is ambiguous. Choose the semantic relation:
- ownership/control: `Own`
- physical containment: `Contains`
- structural part: `PartOf`
- group/set membership: `MemberOf`
- attribute possession: an attribute predicate such as `Color` or `EyeColor`

Do not conflate these into a generic `Has` unless the distinction truly does not matter.

### Events, passive voice, and argument alternations

Use the same event-role structure for active and passive voice. “Sam broke the window” and “The window was broken by Sam” should map to the same event type and roles.

If the agent is omitted in a passive, leave the agent unasserted or use an unknown context-specific placeholder only if the discourse requires an entity.

Inchoative alternations such as “The door opened” describe a change of state without necessarily asserting an external agent.

### Change of state, creation, and destruction

Represent the event and the resulting state separately. For creation/destruction, assert the creation/destruction event and the created/destroyed entity relation or resulting existence/status predicate.

Do not infer current nonexistence from “destroyed” unless the domain model includes that rule.

### Transfer, communication, and exchange

Use event roles such as `Agent`, `Recipient`, `Theme`, `Source`, and `Goal`. For communication, distinguish:
- the communication event;
- the speaker/source;
- the addressee;
- the content representation.

Do not assert reported content as true unless the construction/source warrants it.

### Motion, path, and spatial relations

For motion, use event roles for mover/theme, source, goal, path, and location. For static spatial relations, use direct predicates such as `In`, `On`, `Under`, `Above`, `Near`, `NorthOf`, or `NextTo`.

Resolve deictic terms such as “here,” “there,” “nearby,” and “closer” relative to a context-specific viewpoint, speaker, or anchor.

### Time, tense, aspect, duration, and ordering

Do not encode tense in predicate names. Use temporal predicates/roles such as `Time`, `StartTime`, `EndTime`, `Duration`, `Before`, `After`, `During`, or `Simultaneous`.

Past tense usually indicates that the event time precedes the utterance time. Future tense often indicates prediction, plan, schedule, or modality; do not assert actual occurrence unless the English entails it.

Progressive/perfect/aspectual forms should be represented with event/state status predicates when relevant, such as ongoing, completed, prior, still-active, or already-completed.

### Repetition, frequency, and habituals

For repeated concrete events, create separate event ids or a count/aggregate fact. For habituals and generics, use rules or frequency predicates. Do not create predicates like `RangThreeTimes` or `UsuallyLateTrain`.

### Negation and absence

Sentential negation should not be mapped to `Not` unless there is an existing proposition whose truth value is being negated in a rule premise.

For lexical negation:
- use an antonym predicate when it is a real state, such as `Closed`;
- use a negative lexical predicate only when it is a primitive domain category, such as `Nonviolent`;
- otherwise represent the positive predicate with low strength or use an explicit absence/exclusion predicate.

Negative quantifiers such as “no,” “none,” and “nobody” require explicit count/absence modeling or negative evidence; they are not licensed by failure to find a proof.

### Modality, obligation, permission, and directives

Do not assert the embedded event as actual merely because it is possible, necessary, permitted, forbidden, intended, or requested.

Use domain predicates such as:
- `Able`
- `Possible`
- `Necessary`
- `Obligated`
- `Permitted`
- `Forbidden`
- `Requested`
- `Intends`
- `Plans`

For imperatives, represent a directive/request/obligation unless the sentence reports that the action occurred.

### Belief, knowledge, reports, and factivity

For non-factive attitudes such as “believes,” “suspects,” “claims,” or “says,” represent the attitude/report relation without automatically asserting the embedded content.

For factive verbs such as “knows,” “realized,” or “regretted,” assert the embedded content separately only if the parser accepts the presupposition as part of the KB.

For reported or evidential information, lower confidence or add source predicates.

### Conditionals, unless, only-if, and counterfactuals

Ordinary “if P then Q” maps to an implication rule.

“P only if Q” maps as P implies Q.

“Unless Q, P” requires a representation of explicit non-Q or absence of Q; do not use negation-as-failure.

Counterfactuals should not be asserted as ordinary factual rules unless the KB is intentionally modeling hypothetical reasoning. Use a custom counterfactual/conditional relation if they must be stored.

### Comparatives, superlatives, ranking, and scalar change

For degree comparison, expose the measured values/distributions and use general comparison rules. “Less than” can usually be represented by reversing the arguments of a greater-than comparison.

Superlatives and rankings require a comparison set plus aggregation or pairwise comparison rules. Do not encode the winner/rank into the predicate name.

Comparative correlatives such as “the more X, the more Y” require an explicit rule relating two measured quantities or changes.

### Plurality, mass nouns, collective, and distributive readings

For distributive readings, assert or query facts about each member. For collective readings, create a group entity and assert the collective predicate about that group.

Mass nouns should be represented as substances/quantities/portions when countability matters.

### Definiteness, indefiniteness, anaphora, and demonstratives

A definite description should resolve to an existing entity when context supplies one. If it does not, create a context-specific entity only if the discourse presupposes one.

An indefinite assertion usually introduces a new witness. A nonspecific indefinite inside a modal or desire should not be treated as a concrete existing entity unless the reading is specific.

Demonstratives such as “this” and “that” require context-specific entity resolution.

### Relative clauses, appositives, and complements

Restrictive relatives add conditions on the same entity. Nonrestrictive relatives and appositives add separate facts.

Complement clauses under attitudes, speech, modality, and factive verbs must respect the embedding predicate’s semantics; do not flatten them blindly into asserted facts.

### Control and raising

Resolve the unspoken subject of control constructions:
- “Dana tried to leave” links Dana to the leaving event.
- “Omar persuaded Lina to stay” links Lina to the staying event.

Raising constructions such as “Maya seems to understand” usually express evidential/modality about the embedded content, not a separate action of seeming by Maya.

### Preposition polysemy

Map prepositions by meaning, not surface form:
- spatial “on” → support/location relation;
- temporal “on Monday” → time relation;
- reliance “on a friend” → dependency/support relation;
- instrument “with a knife” → instrument role;
- accompaniment “with Nora” → companion/co-participant relation.

### Focus particles, presuppositions, and exceptions

“Only,” “except,” “besides,” and similar constructions often require closed-world, exclusion, or cardinality modeling. Represent exceptions explicitly rather than hiding them in predicate names.

“Also,” “too,” and “even” are often discourse/pragmatic markers; encode only their truth-conditional contribution unless the focus information matters.

Presupposition triggers such as “again,” “stop,” “still,” and definite descriptions should add presupposed facts only when the system is intended to accept those presuppositions.

### Non-intersective and intensional adjectives

Do not treat these as ordinary intersective properties:
- “former mayor” means prior mayor status, not current `Mayor`.
- “fake diamond” should not assert `Diamond`.
- “alleged spy” should not assert `Spy` without source/evidence qualification.

Use predicates such as `Former`, `Fake`, `Alleged`, or source-linked claim structures as appropriate.

### Inclusion, exclusion, and set membership

Use `MemberOf`, `InSet`, `IncludedIn`, or domain-specific membership predicates for inclusion.

Use `ExcludedFrom`, `NotMemberOf`, low-strength membership, or explicit count/absence modeling for exclusion, depending on the intended semantics.