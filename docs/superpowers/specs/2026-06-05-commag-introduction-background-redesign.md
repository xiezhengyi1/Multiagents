# IEEE Communications Magazine Sections I-II Redesign

## Objective

Rebuild the first two sections of `docs/main.tex` around one argument: CoreAgents is a controlled transition architecture between AI-assisted operation in today's core network and a future AI-native core network.

## One-Sentence Argument

In intent-driven core-network policy control, CoreAgents introduces a staged multi-agent reasoning layer and a deterministic enforcement layer that preserve existing network-function authority while adding the semantic, context-management, and trace capabilities needed for gradual evolution toward AI-native operation.

## Claim Boundary

- The article does not claim that CoreAgents is an AI-native core network.
- The article does not claim that multi-agent decomposition eliminates context growth; it limits model-visible context and reduces cross-stage error coupling.
- The article does not claim that deterministic checks guarantee complete safety; they establish explicit, auditable safety and reliability boundaries.
- LLM agents do not replace AMF, SMF, PCF, standardized signaling procedures, or interface contracts.

## Section I: Introduction

The Introduction follows an application-first evolutionary bridge structure.

1. **6G setting and concrete pressure.** Open with an emergency mobility corridor in which remote driving, telemedicine, drones, robots, and AR services compete for policy and slice resources.
2. **Two complementary evolution paths.** Present AI-assisted networking as intelligence added to existing functions and workflows, and AI-native networking as intelligence, data, computing, and networking designed together. Treat these as analytical paths synthesized from References 1-3, not as a universally standardized binary taxonomy.
3. **Transition gap.** Explain why a direct jump is unsafe and impractical: legacy deployment, standardized network-function authority, strict interfaces, and telecom reliability requirements remain binding.
4. **Technical bottlenecks.** Identify two coupled problems: a monolithic agent accumulates heterogeneous intent, topology, policy, tool, and retry context; unconstrained probabilistic outputs cannot be treated as executable control decisions.
5. **Present article.** Position CoreAgents as a controlled agentic transition layer. Stage-specific agents handle interpretation, grounding, planning, and diagnosis; deterministic modules handle mediation, compilation, dispatch, and assurance.
6. **Contributions.** State bounded contributions in positioning, context-partitioned multi-agent control, deterministic enforcement, and scenario-based evaluation.

## Section II: Evolution Path and Design Principles

### A. From AI Assistance to AI-Native Networking

Define the two endpoints and place CoreAgents between them. The bridge preserves current functions and interfaces while producing intent artifacts, policy artifacts, and assurance traces useful for later native intelligence.

### B. Where Agents Can Reside in the Core Network

Discuss four candidate placements:

- management and orchestration systems, including intent-management and OSS functions;
- service exposure and application-facing control;
- implementation inside a network function, such as policy decision support in a PCF;
- a cross-function policy-assistance layer above standardized execution interfaces.

Select the fourth placement for CoreAgents because it provides cross-function semantic reasoning without giving an LLM direct authority over signaling procedures.

### C. Why a Multi-Agent Pipeline

Explain that policy control combines heterogeneous context classes with different lifetimes and owners. A single agent must repeatedly carry business vocabulary, object bindings, network state, constraints, tool observations, and retry history. CoreAgents partitions this context by responsibility and passes only typed artifacts downstream. The rationale is context isolation, fault localization, and selective repair, not anthropomorphic collaboration.

### D. Why Deterministic Enforcement Remains Necessary

Separate proposal from authority. Agents propose structured intents and plans; deterministic modules validate identifiers and schemas, mediate conflicts, compile network-facing objects, dispatch through authorized adapters, and evaluate outcomes. Failures are classified and returned to the responsible stage.

### E. Design Principles

Use six principles:

1. bounded autonomy;
2. context and responsibility isolation;
3. evidence-backed grounding;
4. typed artifact handoffs;
5. deterministic enforcement and observable receipts;
6. failure-directed repair.

## Terminology Ledger

- **AI-assisted networking:** AI augments existing network functions, management workflows, and operator decisions.
- **AI-native networking:** intelligence, data, models, computing, and network functions are architectural design elements rather than external add-ons.
- **controlled agentic transition layer:** the article's preferred description of CoreAgents' architectural position.
- **multi-agent pipeline:** stage-specific agents connected through typed artifacts; avoid describing it as free-form agent conversation.
- **deterministic enforcement layer:** mediation, validation, compilation, dispatch, and assurance components that retain execution authority.
- **context growth / context overload:** preferred over an absolute claim of "context explosion" unless used informally and then defined.

## Evidence Alignment

- References 1-3 support the evolution toward agent-assisted autonomous operation, AI-enabled 6G capabilities, trustworthiness, smooth migration, and increasingly native architectural integration.
- Repository implementation supports context projection, token-aware context policies, typed artifacts, policy guards and compilers, dispatch, assurance evaluation, and directed diagnosis/retry.
- Experimental claims remain limited to the scenarios and metrics reported later in the manuscript.

## Out of Scope

- Figures and captions are unchanged in this revision.
- The abstract, later architecture sections, experiments, and conclusion are not rewritten except where a citation key must be corrected for compilation consistency.
- No claim is made about production certification or formal safety guarantees.
