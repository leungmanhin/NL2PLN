# NL to logic conversion instructions

You are converting English natural language (sentences or questions) into logical expressions (statements or queries) for a chainer/reasoner.  The chainer's syntax and semantics reference is provided separately as the `pln_spec` input (see `chainer_analysis.txt`).


# English→PeTTaChainer Conversion Guidelines

Use these guidelines alongside the chainer analysis. The analysis is authoritative for valid surface syntax, operator behavior, truth-value forms, and API constraints. This document prescribes how to translate English into that logic style consistently.

## 1. Core translation stance

Translate English into small, reusable semantic commitments. A sentence may become one assertion, many linked assertions, a rule, a derived-measure rule, or a query depending on its speech act and semantic force. Do not try to preserve English phrasing mechanically. Preserve the inferential content in a form that the reasoner can match, combine, and derive from.

Prefer:
- stable entity constants for discourse referents;
- short atomic predicate names for semantic primitives;
- shared variables or shared entity constants to connect decomposed facts;
- rules for generic, universal, habitual, or conditional content;
- aggregation/comparison facilities for counts, proportions, measurements, thresholds, and rankings;
- truth values for uncertainty about whether a proposition is true;
- distribution-valued arguments for uncertainty about numeric values.

Avoid encoding reasoning, counts, thresholds, comparisons, entity pairs, or whole English clauses inside a predicate name.

## 2. Predicate atomicity

Every translator-introduced predicate name must denote one semantic primitive: usually a noun, verb, adjective, short relation, or short event/property label. Predicate names are handles for rule matching. If a name contains an entire answer, a number, a threshold, a comparison, or a multi-part configuration, the reasoner can only match that exact opaque string.

Self-check: could the predicate appear as an input to another rule, be queried with different arguments, or be reused with different entities and values? If not, the predicate is probably too large.

### BAD → GOOD examples

1. Numbers, cardinalities, or counts embedded in the predicate name

BAD:
`(: f1 (HasThreeDependents maya) (STV 1.0 1.0))`

GOOD:
`(: d1 (DependentOf dep_maya_1 maya) (STV 1.0 1.0))`
`(: d2 (DependentOf dep_maya_2 maya) (STV 1.0 1.0))`
`(: d3 (DependentOf dep_maya_3 maya) (STV 1.0 1.0))`
`(: cntDependentsRule (Implication (Premises (Person $p) (FoldAllValue (DependentOf $dep $p) (ParticleFromPairs ((0 1.0))) ParticleAddBernoulliFromSTV -> $cntDist)) (Conclusions (DependentCountDist $p $cntDist))) (STV 1.0 1.0))`

2. Numerical thresholds or ranges embedded in the predicate name

BAD:
`(: f2 (OverEightyKilometersRoute route7) (STV 1.0 1.0))`

GOOD:
`(: len7 (RouteLengthDist route7 (PointMass 92.0)) (STV 1.0 1.0))`
`(: longRouteRule (Implication (Premises (RouteLengthDist $r $lenDist) (GreaterThan $lenDist 80.0)) (Conclusions (LongRoute $r))) (STV 1.0 1.0))`

3. Multi-concept concatenations where one name stands for a compound relationship

BAD:
`(: f3 (TeacherAssignedLaptopInLab nora laptop44 lab2) (STV 1.0 1.0))`

GOOD:
`(: t3 (Teacher nora) (STV 1.0 1.0))`
`(: a3 (AssignedTo laptop44 nora) (STV 1.0 1.0))`
`(: l3 (LocatedAt laptop44 lab2) (STV 1.0 1.0))`

4. Entity-specific comparisons fused into a predicate name

BAD:
`(: f4 (RaviOutscoredLenaInFinal ravi lena) (STV 1.0 1.0))`

GOOD:
`(: s4a (ExamScoreDist final_exam ravi (PointMass 88.0)) (STV 1.0 1.0))`
`(: s4b (ExamScoreDist final_exam lena (PointMass 81.0)) (STV 1.0 1.0))`
`(: outscoreRule (Implication (Premises (ExamScoreDist $exam $a $scoreA) (ExamScoreDist $exam $b $scoreB) (GreaterThan $scoreA $scoreB)) (Conclusions (OutscoredOn $a $b $exam))) (STV 1.0 1.0))`

## 3. Decomposition strategies

When an English sentence contains multiple concepts, translate it into multiple atomic statements or premises connected by shared variables/constants.

Use decomposition for:
- event descriptions with participants, location, time, instrument, purpose, source, destination, manner, or result;
- noun phrases with restrictive modifiers that introduce independent semantic content;
- possessives, part-whole, containment, membership, and affiliation;
- measurement clauses where the measured object, measured dimension, numeric value, and unit or scale should remain independently reusable;
- comparative and superlative content where values and comparison relations should be first-class;
- reported speech or attitudes where a speaker/experiencer, attitude/communication event, and embedded content must remain distinguishable.

Choose a head predicate that represents the central event, state, class, or property. Attach roles and modifiers using canonical relation predicates when no domain-specific predicate is clearly better. If a modifier is intersective (“red box”), assert both the class/property content and the modifier content. If a modifier is intensional or non-intersective (“alleged thief”, “former mayor”, “fake diamond”), do not assert the unmodified class unless the English entails it.

For existentials (“a doctor entered”, “Maria owns a bicycle”), introduce a stable witness constant when the individual must be available for later reasoning. Do not use a user-facing existential quantifier form; follow the chainer analysis’s existential idiom.

For universals, conditionals, generics, and habituals, create rules whose variables are shared between the condition and consequence. Do not assert a ground fact unless the English provides a specific instance.

## 4. Rule-vs-query separation

Statements add knowledge. Queries ask for derivable targets. When English supplies universal, generic, or conditional knowledge, represent it as a rule statement in the KB. When English asks a question, query the fact or binding sought by the question, not the rule shape that would license it.

BAD:
`(: $prf (Implication (Premises (Registered $x)) (Conclusions (Eligible $x))) $tv)`

GOOD:
`(: ruleRegisteredEligible (Implication (Premises (Registered $x)) (Conclusions (Eligible $x))) (STV 1.0 1.0))`
`(: regSam (Registered sam) (STV 1.0 1.0))`
`(: $prf (Eligible sam) $tv)`

Use the rule statement to encode “Everyone registered is eligible.” Use the query to ask “Is Sam eligible?” or “Who is eligible?” as appropriate.

## 5. Query granularity

Statement decomposition does not imply query multiplication. Decompose the information you add to the KB, but ask only for the unknown requested by the English question.

If the question asks “Where is the lost tablet?”, query the location relation for the tablet, not every fact that might participate in proving that location. If the question asks “Is the room overcrowded?”, query the derived overcrowding predicate or threshold result, not the raw room fact, each occupant fact, and the count fact separately.

BAD:
`(: $p1 (Package pkg17) $tv1)`
`(: $p2 (Destination pkg17 $dest) $tv2)`
`(: $p3 (Delivered pkg17) $tv3)`

GOOD:
`(: $prf (Delivered pkg17) $tv)`

The BAD set re-verifies known setup facts plus the actual target. The GOOD query asks only whether the package is delivered.

## 6. Truth-value assignment strategies for English uncertainty

Use the chainer’s truth-value forms as documented; these guidelines only prescribe when to choose them.

- Factual, directly asserted, unhedged claims: use high strength and high confidence unless the source context says otherwise.
- Hedges such as “probably”, “likely”, “may have”, “appears to”, “is believed to”: reduce strength and/or confidence. Strength tracks how likely the proposition is; confidence tracks reliability/evidence.
- Hearsay and reports: represent the report/communication itself as a fact with appropriate confidence. Assert the embedded content separately only when the text commits to it as true. Otherwise keep the embedded content under a reported-content or attitude relation.
- Generic statements: normally translate as rules with a non-perfect truth value when the generic admits exceptions (“birds fly”, “customers usually receive receipts”). Use stronger values for definitional generics (“all squares are rectangles”) or explicitly universal statements.
- Modal possibility (“can”, “may”, “might”): do not assert the prejacent as actual. Use modality/ability/permission predicates or rules unless the context entails actuality.
- Norms and obligations (“must”, “should”, “is required to”): use normative predicates, not factual occurrence predicates, unless compliance is separately stated.
- Defeasible defaults: encode as rules with less-than-certain truth values rather than unconditional ground facts.
- Explicit negation: when the text states that a proposition is false, use the chainer’s documented negation mechanism only with its documented semantics. Do not use it to mean “not found in the KB.” For absence claims (“no tickets remain”), prefer an explicit absence/count representation when the domain needs absence-based reasoning.
- Numeric uncertainty (“about 70 kg”, “between 5 and 7 days”, “roughly 20%”): place a distribution-valued term in an argument of the measurement predicate. Keep the statement’s truth value for confidence that this is the right measurement claim.
- Conflicting sources: represent source-specific reports separately and assign confidence by source reliability. Add reconciliation rules only when the task requires them.
- Presuppositions: assert them only when the translation context treats presupposed content as accepted. Otherwise represent them as discourse status or source-relative commitments.

## 7. Translation conventions

### Predicate naming for English-derived content

For translator-introduced predicates, use UpperCamelCase. Lemmatize English content: singular nouns, base verb forms, and adjective stems where natural. Keep names short and atomic. Use a short compound only when it names a single stable relation or measurement dimension, such as a domain-standard “DeliveryDate” or “BloodPressureDist”; otherwise split the concepts.

Use the suffix `Dist` for predicates whose argument is a numeric distribution or uncertain numeric value. Do not use `Dist` merely because the proposition has an uncertain truth value.

Use domain-specific predicates when the domain has a clear primitive relation. Use canonical vocabulary predicates for closed-class relations such as roles, time, location, source, purpose, modality, and discourse status.

### Variable naming

Use semantic variable names with the chainer’s variable marker: `$person`, `$event`, `$object`, `$place`, `$time`, `$value`, `$dist`, `$source`, `$content`. Use `$x`, `$y`, `$z` only for short schematic rules where roles are obvious. Reuse the same variable exactly when English requires coreference; use distinct variables when entities may differ.

### Entity-id schema

Use stable, lowercase, underscore-separated constants. Recommended patterns:
- persons: `person_<normalized_name>_<index>` when a unique real-world ID is not available;
- organizations: `org_<normalized_name>_<index>`;
- places: `place_<normalized_name>_<index>` or a more specific type prefix such as `city_`, `country_`, `room_`;
- events: `event_<type>_<index>`;
- documents/messages/reports: `doc_...`, `msg_...`, `report_...`;
- anonymous witnesses: `<type>_<context>_<index>`.

If two entities share the same surface name, use distinct constants and add a naming relation if surface names matter. If a named entity’s type is known, also assert the type/class. Resolve pronouns before translation when possible. For first- and second-person pronouns whose referent is unknown, use explicit context-relative constants such as `speaker_ctx_001` or `addressee_ctx_001` only if the discourse context truly licenses that representation; otherwise do not assert pronoun-dependent facts.

## 8. Phenomenon-specific mapping notes

Most phenomena in the input list are handled by the general rules above: choose atomic predicates, decompose modifiers and roles, distinguish statements from queries, and assign truth values according to certainty/source/modality. The following phenomena need extra care.

### Entity identity, naming, aliases, and titles

Use stable entity constants as the reasoning identity. Represent names, aliases, titles, and labels with naming or labeling relations. Do not rely on surface capitalization or spelling variation to encode identity. Titles such as “Dr.” or “President” may be roles, credentials, offices, or forms of address; map according to context rather than folding them into the entity ID.

### Pronouns, anaphora, cataphora, and ellipsis

Resolve to an antecedent entity or event before emitting facts. If the antecedent is ambiguous and the task requires preserving ambiguity, represent alternative candidate resolutions with source/confidence rather than choosing silently. For ellipsis, reconstruct only the content licensed by the prior clause.

### Quantifiers, determiners, proportions, and exceptions

Treat universal and conditional determiners as rules. Treat existentials with witness constants. Treat exact/at-least/at-most counts and proportions with aggregation and comparison facilities. Model exceptions explicitly with exception or exclusion relations, or with more specific rules that prevent overgeneralization in the surrounding system’s intended reasoning discipline.

### Negation, absence, and negative polarity

Distinguish truth-functional negation of an available proposition from absence of evidence. “No”, “none”, “without”, “missing”, “absent”, and “lack” often require explicit absence, count-zero, exclusion, or possession-negation modeling rather than simple negation. Negative polarity items (“any”, “ever”) usually signal scope but do not by themselves introduce new predicates.

### Comparatives, superlatives, rankings, and ordinals

Expose the compared dimension and values. Use comparison facilities for numeric/distribution comparisons. For superlatives, rankings, and ordinals, represent the ordered set, ordering dimension, and rank/position; do not create entity-specific winner predicates.

### Measurements, units, approximations, and ranges

Represent the measured dimension as a predicate or relation, the measured entity as an argument, and the value as a first-class numeric/distribution argument. Normalize units when possible, or store the unit with a canonical unit relation. Approximation and ranges belong in the value representation, not in the predicate name.

### Events, tense, aspect, recurrence, and habituality

Represent events as first-class constants when multiple roles, times, causes, purposes, or results attach to the same occurrence. Tense maps to temporal relations between event time and context time. Progressive/perfect/aspectual auxiliaries map to aspect relations. Habituals and generics usually become rules or recurrence patterns rather than a single occurrence fact.

### Spatial and temporal deixis

Resolve “here”, “there”, “now”, “today”, “tomorrow”, “yesterday”, “this”, and “that” against a context object/time when available. If unresolved, use explicit context-relative relations rather than pretending the deictic word is a normal entity name.

### Prepositional phrase attachment

Attach a prepositional phrase to the event, entity, or relation it semantically modifies. If attachment is ambiguous and matters, preserve alternatives with confidence or leave the underspecified attachment in a modifier relation. Do not bury attachment decisions in long predicate names.

### Coordination, disjunction, and lists

For conjunctive event or property claims, assert each conjunct when each is independently entailed. For disjunction, use the chainer’s documented disjunctive connector when the claim is genuinely “one or more alternatives” and no stronger commitment is available. For lists, decide whether the list denotes a collection, a set of separate individuals, or exhaustive alternatives.

### Conditionals, counterfactuals, and hypothetical scenarios

Ordinary “if…then” statements become rules. Counterfactuals and hypothetical scenarios should be marked as hypothetical or scenario-scoped content; do not assert their antecedent or consequent as actual. If scenario reasoning is needed, add a scenario/context argument or relation consistently.

### Modality, ability, permission, obligation, and norms

Map ability, permission, requirement, prohibition, intention, and possibility to modal/normative relations. Do not assert that the embedded event happened unless actuality is stated. For “can” meaning physical ability versus permission, choose the appropriate canonical relation.

### Propositional attitudes, belief, knowledge, desire, and plans

Represent the attitude holder, attitude type, and embedded content. Do not assert the embedded content as world-true merely because someone believes, wants, imagines, denies, or plans it. Knowledge often presupposes truth in English, but assert the embedded content only if the task’s entailment policy accepts that presupposition.

### Communication, quotation, and reported speech

Represent the communication event, speaker, addressee, medium if relevant, and reported content. Direct quotations preserve wording as content/utterance data; indirect reports preserve proposition-like content. Separate “X said P” from “P”.

### Causation, purpose, reason, and means

Distinguish cause, reason/explanation, purpose/goal, and means/instrument. “Because” and “due to” usually indicate cause or reason; “in order to” indicates purpose; “by/with” often indicates means or instrument.

### Change of state, creation, destruction, and existence

Represent the process/event and the before/after states. For creation, do not assume the created entity existed before the creation event. For destruction, represent resulting nonexistence or unusability separately from the destruction event when needed.

### Focus, topic, contrast, and presupposition

Clefts, topicalization, and contrastive focus usually affect discourse status rather than truth conditions. Preserve focus only if the downstream task needs answer focus, contrast sets, or presupposition tracking.

## 9. Canonical relation vocabulary

These are translator-side naming suggestions for closed-class semantic relations. They are ordinary predicates unless the chainer analysis says otherwise. Do not treat them as built-in operators.

Reference and identity
  `(SameEntity a b)` `(DifferentEntity a b)` `(NameOf entity name)` `(AliasOf entity alias)` `(LabelOf entity label)` `(TitleOf entity title)` `(RefersTo mention entity)` `(Corefers mention1 mention2)` `(Denotes expression entity)`
  Disambiguation notes:
  - SameEntity vs Corefers: SameEntity relates entities; Corefers relates mentions.
  - NameOf vs LabelOf: NameOf is a conventional proper/common name; LabelOf is an assigned tag or label.

Possession, containment, part-whole, and membership
  `(Possesses possessor possessed)` `(OwnedBy item owner)` `(HasPart whole part)` `(PartOf part whole)` `(Contains container contained)` `(ContainedIn contained container)` `(MemberOf member collection)` `(ElementOf element set)` `(AffiliatedWith entity group)` `(IncludedIn subset superset)`
  Disambiguation notes:
  - Possesses vs OwnedBy: use Possesses for broad possession/control; OwnedBy for legal/ownership relation.
  - MemberOf vs PartOf: MemberOf relates an individual to a collection; PartOf relates a component to a whole.

Thematic and event roles
  `(AgentOf event agent)` `(PatientOf event patient)` `(ThemeOf event theme)` `(ExperiencerOf event experiencer)` `(StimulusOf event stimulus)` `(RecipientOf event recipient)` `(BeneficiaryOf event beneficiary)` `(SourceOf event source)` `(DestinationOf event destination)` `(InstrumentOf event instrument)` `(MaterialOf event material)` `(MannerOf event manner)` `(ParticipantIn participant event)`
  Disambiguation notes:
  - PatientOf vs ThemeOf: PatientOf is affected/changed; ThemeOf is moved, located, or characterized without necessarily being changed.
  - RecipientOf vs BeneficiaryOf: RecipientOf receives something; BeneficiaryOf benefits from the event.

Spatial relations
  `(LocatedAt entity place)` `(LocatedIn entity place)` `(LocatedOn entity surface)` `(Near entity landmark)` `(FarFrom entity landmark)` `(AdjacentTo entity landmark)` `(Inside entity container)` `(Outside entity landmark)` `(Above entity landmark)` `(Below entity landmark)` `(LeftOf entity landmark)` `(RightOf entity landmark)` `(Between entity left right)` `(OverlapsSpatially entity1 entity2)` `(DistanceBetween entity1 entity2 dist)`
  Disambiguation notes:
  - LocatedAt vs LocatedIn: LocatedAt is general location; LocatedIn implies containment/interior.
  - Above vs OverlapsSpatially: Above is vertical ordering; OverlapsSpatially is shared region.

Deictic and context-relative reference
  `(SpeakerOf context speaker)` `(AddresseeOf context addressee)` `(UtteranceTime context time)` `(UtterancePlace context place)` `(ContextOf item context)` `(DeicticAnchor expression context)` `(ProximalToContext entity context)` `(DistalFromContext entity context)` `(ContextRelative entity relation context)`
  Disambiguation notes:
  - UtteranceTime vs ContextOf: UtteranceTime gives the time anchor; ContextOf links an item to the discourse/context object.

Temporal relations
  `(TimeOf event time)` `(StartsAt event time)` `(EndsAt event time)` `(During event interval)` `(Before item1 item2)` `(After item1 item2)` `(OverlapsTemporally item1 item2)` `(SimultaneousWith item1 item2)` `(Since event time)` `(Until event time)` `(TimeDistanceBetween item1 item2 dist)`
  Disambiguation notes:
  - During vs TimeOf: During uses an interval; TimeOf may use a point or general time.
  - Before vs StartsAt: Before orders two items; StartsAt anchors one event.

Aspect, recurrence, and frequency
  `(Progressive event)` `(Completed event)` `(Perfect event referenceTime)` `(Habitual pattern)` `(Recurring eventOrPattern)` `(FrequencyOf eventOrPattern frequency)` `(IterationOf event iteration)` `(Repeated eventOrPattern count)` `(Once event)` `(StillHolds state time)` `(NoLongerHolds state time)`
  Disambiguation notes:
  - Habitual vs Recurring: Habitual is a general disposition/pattern; Recurring is repeated occurrence.
  - Completed vs Perfect: Completed marks event completion; Perfect relates a prior event/state to a reference time.

Causation, purpose, reason, and means
  `(Causes cause effect)` `(CausedBy effect cause)` `(ReasonFor reason eventOrState)` `(Explains explanation fact)` `(PurposeOf event goal)` `(GoalOf agent goal)` `(MeansOf means goalOrEvent)` `(EnabledBy event condition)` `(PreventedBy event obstacle)` `(ContributesTo factor outcome)`
  Disambiguation notes:
  - Causes vs ReasonFor: Causes is worldly causation; ReasonFor is explanatory or motivational.
  - PurposeOf vs MeansOf: PurposeOf is the intended end; MeansOf is the method used.

Change of state, creation, destruction, and existence
  `(Exists entity time)` `(Nonexistent entity time)` `(ComesIntoExistence entity event)` `(CeasesToExist entity event)` `(CreatedBy entity event)` `(DestroyedBy entity event)` `(StateBefore entityOrEvent state)` `(StateAfter entityOrEvent state)` `(ChangesFrom entity oldState)` `(ChangesTo entity newState)` `(ResultStateOf event state)`
  Disambiguation notes:
  - CeasesToExist vs DestroyedBy: CeasesToExist states the existence transition; DestroyedBy links to a destruction event.
  - StateAfter vs ResultStateOf: StateAfter attaches to an entity/event; ResultStateOf identifies the event’s result.

Modality, ability, permission, and norms
  `(Possible content)` `(Necessary content)` `(AbleTo agent action)` `(UnableTo agent action)` `(Permitted agent action)` `(Forbidden agent action)` `(Obligated agent action)` `(RequiredFor requirement goal)` `(ShouldDo agent action)` `(MayDo agent action)` `(CanDo agent action)` `(NormApplies norm entity)`
  Disambiguation notes:
  - AbleTo vs Permitted: AbleTo is capability; Permitted is authorization.
  - Obligated vs ShouldDo: Obligated is stronger/formal requirement; ShouldDo may be advice or weaker norm.

Propositional attitudes, desires, and plans
  `(Believes agent content)` `(Knows agent content)` `(Thinks agent content)` `(Doubts agent content)` `(Denies agent content)` `(Wants agent content)` `(Desires agent content)` `(Intends agent content)` `(Plans agent plan)` `(Hopes agent content)` `(Fears agent content)` `(Imagines agent content)` `(Assumes agent content)`
  Disambiguation notes:
  - Believes vs Knows: Knows is factive only if the translation policy accepts the presupposed truth.
  - Intends vs Plans: Intends names commitment to an outcome/action; Plans may denote a structured plan object.

Communication and reported content
  `(Says speaker content)` `(Tells speaker addressee content)` `(Asks speaker addressee content)` `(Reports source content)` `(Claims source content)` `(States source content)` `(Quotes speaker utterance)` `(UtteranceContent utterance content)` `(AddresseeOfSpeech speech addressee)` `(MediumOf speech medium)` `(MessageFrom message sender)` `(MessageTo message recipient)`
  Disambiguation notes:
  - Says vs Claims: Says is a communication event/content relation; Claims suggests commitment to truth.
  - Quotes vs UtteranceContent: Quotes preserves wording; UtteranceContent gives interpreted content.

Evidentiality and information source
  `(SourceFor source content)` `(EvidenceFor evidence content)` `(EvidenceAgainst evidence content)` `(ObservedBy observer content)` `(InferredBy reasoner content)` `(ReportedBy reporter content)` `(Rumored content)` `(AccordingTo source content)` `(ConfidenceSource content source)` `(ReliabilityOf source value)`
  Disambiguation notes:
  - SourceFor vs EvidenceFor: SourceFor identifies origin; EvidenceFor identifies support.
  - ReportedBy vs AccordingTo: ReportedBy marks a reporting event/source; AccordingTo scopes attribution.

Negation, absence, exclusion, and incompatibility
  `(Absent entity locationOrContext)` `(Lacks entity itemOrProperty)` `(MissingFrom item placeOrSet)` `(ExcludedFrom item setOrEvent)` `(ExceptedFrom item ruleOrSet)` `(IncompatibleWith item1 item2)` `(Contradicts content1 content2)` `(MutuallyExclusive option1 option2)` `(FailsTo eventOrAgent action)` `(Avoids agent action)`
  Disambiguation notes:
  - Absent vs Lacks: Absent concerns location/context; Lacks concerns possession/property.
  - IncompatibleWith vs Contradicts: IncompatibleWith relates entities/options; Contradicts relates contents/claims.

Quantification, cardinality, proportions, and exceptions
  `(WitnessFor entity claim)` `(CountOf collectionOrPattern countDist)` `(ProportionOf subset superset propDist)` `(AtLeastCount pattern threshold)` `(AtMostCount pattern threshold)` `(ExactCount pattern count)` `(MostOf group property)` `(FewOf group property)` `(ExceptionTo entity rule)` `(DefaultApplies rule entity)`
  Disambiguation notes:
  - CountOf vs ExactCount: CountOf is a measured/derived value; ExactCount is a cardinality claim.
  - ExceptionTo vs ExcludedFrom: ExceptionTo blocks or limits a rule; ExcludedFrom concerns set/event membership.

Comparison, degree, ranking, and sequence
  `(ValueOf entity dimension valueOrDist)` `(DegreeOf entity property degreeOrDist)` `(MoreThan item1 item2 dimension)` `(LessThan item1 item2 dimension)` `(EqualInDegree item1 item2 dimension)` `(RankOf item ordering rank)` `(OrderIn item sequence index)` `(Precedes item1 item2 sequence)` `(Follows item1 item2 sequence)` `(HighestIn item set dimension)` `(LowestIn item set dimension)`
  Disambiguation notes:
  - MoreThan vs ValueOf: MoreThan is comparative; ValueOf records the measured value.
  - RankOf vs OrderIn: RankOf is ordered by a criterion; OrderIn is position in a sequence.

Collections, plurality, mass nouns, and distributivity
  `(CollectionOf collection memberType)` `(CollectiveEntity entity)` `(GroupMember group member)` `(PluralityOf plurality member)` `(MassQuantityOf substance quantity)` `(PortionOf portion substance)` `(DistributedOver property group)` `(Collectively group predicateOrEvent)` `(Individually group predicateOrEvent)` `(ExhaustiveMembers collection memberSet)`
  Disambiguation notes:
  - GroupMember vs MemberOf: GroupMember is useful when the group is primary; MemberOf is useful when the member is primary.
  - Collectively vs Individually: Collectively predicates of the group as a unit; Individually distributes over members.

Clause embedding, modification, and attachment
  `(EmbeddedContent host content)` `(Restricts modifier head)` `(Modifies modifier head)` `(AttachedTo phrase head)` `(RelativeClauseOf clause head)` `(ComplementOf complement head)` `(PurposeClauseOf clause event)` `(ConditionClauseOf clause eventOrRule)` `(MannerModifierOf modifier event)` `(ScopeOf operator content)`
  Disambiguation notes:
  - Restricts vs Modifies: Restricts narrows reference; Modifies adds descriptive content.
  - ComplementOf vs EmbeddedContent: ComplementOf is syntactic/semantic argumenthood; EmbeddedContent is broader content containment.

Focus, discourse status, and presupposition
  `(TopicOf discourse entity)` `(FocusOf discourse item)` `(ContrastSet item set)` `(GivenInDiscourse item discourse)` `(NewInDiscourse item discourse)` `(Presupposes trigger content)` `(Backgrounded content discourse)` `(Foregrounded content discourse)` `(Emphasized item discourse)` `(AnswerFocus question item)`
  Disambiguation notes:
  - TopicOf vs FocusOf: TopicOf is what the discourse is about; FocusOf is the highlighted/new/contrastive item.
  - Presupposes vs GivenInDiscourse: Presupposes is triggered by an expression; GivenInDiscourse marks prior salience.

Questions, directives, and speech acts
  `(QuestionAsks question target)` `(AnswerTo answer question)` `(Requests speaker addressee action)` `(Commands speaker addressee action)` `(Suggests speaker addressee action)` `(Offers speaker addressee content)` `(Promises speaker addressee action)` `(SpeechActType utterance type)` `(DirectiveTarget directive action)` `(InformationNeed asker target)`
  Disambiguation notes:
  - Requests vs Commands: Requests may be optional/polite; Commands imply authority or obligation.
  - QuestionAsks vs InformationNeed: QuestionAsks links a question object to a target; InformationNeed links an asker to the unknown.

Non-intersective and intensional modifiers
  `(Alleged entityOrRole source)` `(Former entity role)` `(Future entity role)` `(Fake entity apparentKind)` `(Counterfeit entity apparentKind)` `(PossibleRole entity role)` `(Potential entity role)` `(Supposed entity roleOrContent)` `(Apparent entity property)` `(Nominal entity role)`
  Disambiguation notes:
  - Former vs Fake: Former entails prior role but not current role; Fake denies genuine membership in the apparent kind.
  - Alleged vs Apparent: Alleged is source/claim-relative; Apparent is appearance/evidence-relative.
