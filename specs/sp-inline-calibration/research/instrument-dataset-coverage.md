# Instrument ↔ Dataset Coverage — Research Summary

## 1. Synthpanel Bundled Instruments

| Instrument | Description | Questions | Question types |
|---|---|---:|---|
| product-feedback | NPS + feature satisfaction + open improvement | 5 | 5 free-text |
| market-research | Competitive preference, purchase intent, pricing sensitivity | 5 | 5 free-text |
| pricing-discovery | Branching probe on pain / pricing / competitors | ~9 | ~3 routing bounded, 6 free-text |
| feature-prioritization | Branching on trade-offs / must-haves / scope | ~9 | ~3 routing, 6 free-text |
| churn-diagnosis | Branching on value / friction / competitive | ~9 | ~3 routing, 6 free-text |
| landing-page-comprehension | Branching on jargon / audience / CTA | ~9 | ~3 routing, 6 free-text |
| name-test | Branching on meaning / pronounceability / memorability | ~9 | ~3 routing, 6 free-text |
| general-survey | Tech / work / daily habits opinion | 5 | 5 free-text |

**Key finding:** All 8 instruments are **100% free-text** (`type: text` in `response_schema`). None declare bounded response_schema (Likert, enum, yes-no) today. Branching instruments route on themes extracted post-hoc by the synthesizer, not on pre-defined enum options.

## 2. SynthBench Datasets

| Dataset | Description | Policy | ~Count | Shape |
|---|---|---|---:|---|
| opinionsqa | 1,498 Pew ATP questions (waves 26-92) | gated | 1,498 | Enum 3-5 options |
| globalopinionqa | 2,556 questions × 138 countries | gated | 2,556 | Likert 0-10, agree/disagree |
| subpop | 3,362 questions × 22 US subpopulations | gated | 3,362 | Enum/Likert aggregated per subpop |
| pewtech | Tech-focused ATP (waves 86-113) | gated | ~200-300 | Enum (tech adoption, privacy, AI) |
| eurobarometer | Consumer modules (EU quarterly/flash) | gated | ~50-100+ | Enum (satisfaction, likelihood, agreement) |
| michigan | UMich Survey of Consumers (13 core monthly) | gated | 13 | Enum 3-option (Good/Pro-Con/Bad) |
| ntia | NTIA Internet Use Survey (binary) | full | ~100+ | Binary Yes/No |
| wvs | WVS Wave 7 (64 countries) | gated | 290+ | Enum/Likert values |
| gss | General Social Survey (1972-present) | full | 500+ | Enum/Likert social attitudes |

## 3. Topical Overlap Matrix

| Synthpanel instrument | SynthBench overlap | Strength | Notes |
|---|---|---|---|
| product-feedback (NPS) | pewtech, globalopinionqa | **MED** | NPS 0-10 → Likert; feature satisfaction → tech opinion enum |
| market-research | pewtech, opinionsqa | LOW | SynthBench minimal on pricing/purchase-intent |
| pricing-discovery | — | **none** | SynthBench is opinion-focused; no willingness-to-pay questions |
| feature-prioritization | pewtech, opinionsqa | LOW | SynthBench rarely asks feature ranking |
| churn-diagnosis | pewtech, globalopinionqa, michigan | MED | michigan "switch" proxy; pewtech tech satisfaction |
| landing-page-comprehension | — | **none** | SynthBench is pure opinion; synthpanel targets comprehension |
| name-test | globalopinionqa, opinionsqa | very weak | Opinions not brand perception |
| general-survey | pewtech, opinionsqa, globalopinionqa, michigan | **STRONG** | Tech attitudes, business/economic outlook, broad opinion |

## 4. Question Type Classification

**SynthBench questions are BOUNDED:**
- **Enum:** OpinionsQA, PewTech, Eurobarometer, NTIA, SubPOP, WVS, GSS, Michigan (3-5 discrete options, fully labeled)
- **Likert:** GlobalOpinionQA, WVS, GSS (0-10 or agreement scales)
- **Binary:** NTIA Internet Use (Yes/No)

**Synthpanel questions are ALL FREE-TEXT:** no bounded response_schema in bundled instruments. Routing in branching instruments uses synthesizer's post-hoc `themes` extraction, not pre-defined enum options.

## 5. Distributional Comparison Feasibility

### Easy (no transformation)

1. **general-survey ↔ pewtech / opinionsqa / globalopinionqa.** Both have open-opinion questions. Extraction schema can categorize synthpanel free responses: `pick_one`, `ranking`, or `likert`. Example: "What trend are you most optimistic about?" → extract enum (technology / economy / health / none) → compare to WVS/OpinionsQA political-economic outlook distributions.
2. **product-feedback NPS ↔ pewtech / globalopinionqa.** Extract NPS score 0-10 from "On a scale of 0-10" text → Likert. Compare directly to globalopinionqa 0-10 scales. JSD on {0-3: dissatisfied, 4-6: neutral, 7-8: satisfied, 9-10: promoters}.

### Moderate (light transformation)

1. **churn-diagnosis ↔ michigan.** Synthpanel "What was broken?" → extract as problem category. Michigan "Good time to buy?" → {Good, Pro-Con, Bad}. Map synthpanel categories to michigan sentiment.
2. **product-feedback feature-satisfaction ↔ pewtech.** Synthpanel "How satisfied with core functionality?" → extract satisfaction level or feature category. PewTech "Adopt this technology?" → enum. Binary/ternary satisfaction bridge.

### Difficult (heavy extraction required)

1. **pricing-discovery ↔ no dataset match.** SynthBench has no willingness-to-pay or pricing-sensitivity questions. Would require free-text price-point extraction → binning (e.g., "$0-50/mo", "$50-200/mo", "$200+") → cannot compare to any dataset.
2. **landing-page-comprehension ↔ no match.** Synthpanel targets comprehension/clarity; SynthBench is opinion/attitude.
3. **name-test ↔ no match.** Brand perception not in SynthBench public-opinion datasets.
4. **feature-prioritization ↔ no match.** SynthBench lacks feature ranking questions.

## 6. Response Schema Inventory

All 8 instruments declare `response_schema` on every question — but every one is `type: text`. No enum/Likert/yes-no schema defined anywhere in bundled instruments.

**Implications:**
- Distributional comparison requires **post-hoc extraction** via synthpanel's `structured/` module
- `--extract-schema` or inline `StructuredOutputConfig(schema=LIKERT_SCHEMA)` converts free text → bounded distribution
- Ground-truth key: `Question.human_distribution` in SynthBench
- Synthetic key: extracted enum/likert distribution from synthpanel response

## 7. Coverage Matrix

| Instrument | opinionsqa | globalopinionqa | pewtech | michigan | eurobarometer | ntia | wvs | gss | subpop |
|---|---|---|---|---|---|---|---|---|---|
| product-feedback NPS | MED | MED | LOW | LOW | LOW | — | LOW | — | LOW |
| market-research | LOW | LOW | LOW | LOW | LOW | — | MED | — | LOW |
| pricing-discovery | — | — | — | — | — | — | — | — | — |
| feature-prioritization | — | — | LOW | — | — | — | — | — | — |
| churn-diagnosis | LOW | MED | MED | MED | LOW | — | MED | — | MED |
| landing-page-comprehension | — | — | — | — | — | — | — | — | — |
| name-test | — | LOW | — | — | — | — | — | — | — |
| general-survey | MED | MED | MED | MED | LOW | LOW | MED | MED | MED |

## 8. Key Transformation Patterns

Where `--extract-schema` fits naturally:
1. **Free-text NPS justification** → LIKERT_SCHEMA (extract 0-10 or map satisfaction sentiment)
2. **Open "what went wrong?"** → PICK_ONE_SCHEMA (extract problem category: value / friction / UX / price / competitor)
3. **"Which feature?"** → RANKING_SCHEMA (extract top-3 from feature list)
4. **"Do you agree?"** → YES_NO_SCHEMA (binary extraction from prose)

**Canonical pairing for inline calibration:**
- Synthpanel question with `response_schema: {type: text}`
- Extraction config `StructuredOutputConfig(schema=LIKERT_SCHEMA)` or `YES_NO_SCHEMA`
- Output `{rating: 7, reasoning: "..."}` → JSD vs SynthBench's `human_distribution: {7: 0.15, 8: 0.28, ...}`

## Coverage map summary

**Sparse but targeted:**
- **3 of 8 instruments** (product-feedback, general-survey, churn-diagnosis) have **weak-to-moderate** topical overlap with SynthBench datasets (primarily pewtech, opinionsqa, globalopinionqa, michigan)
- **5 of 8 instruments** (pricing-discovery, landing-page-comprehension, name-test, feature-prioritization, market-research) have **no meaningful overlap** — they probe product/UX specifics; SynthBench covers public opinion
- All overlap requires **free-text → categorical extraction** via bundled schemas
- **Easiest inline calibration:** general-survey + globalopinionqa/pewtech + LIKERT/PICK_ONE extraction
- **Highest ROI for ground truth:** map general-survey tech-attitude questions to pewtech adoption/privacy/AI questions; LIKERT_SCHEMA extraction → JSD vs pewtech actual distribution
