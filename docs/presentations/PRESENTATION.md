# Where Does Bias Hide?
## Token-Level Attribution Reveals Distributed Bias Propagation in Late-Interaction Retrieval

**Chris Ziyu Kong · Brown University · CSCI 2952W**

---

> This document is a **slide-by-slide presentation deck** with speaker notes.
> Each `---` separator marks a new slide. Speaker notes are in blockquotes below each slide.

---

## Slide 1 — Title

# Where Does Bias Hide?

### Token-Level Attribution Reveals Distributed Bias Propagation in Late-Interaction Retrieval

**Chris Ziyu Kong**

Brown University · CSCI 2952W Critical AI and Data Studies · Spring 2026

> **Speaker Notes:**
>
> Good afternoon everyone. My name is Chris, and today I'm going to talk about a question that sits at the intersection of search engine technology and social fairness: **where, exactly, does bias hide inside a search engine?**
>
> We all know that AI systems can be biased. But saying "it's biased" is like saying "you're sick" without telling you what's wrong or where it hurts. My research is about building a diagnostic tool — think of it as an X-ray machine — that lets us look *inside* a search algorithm and pinpoint exactly which parts are carrying bias.

---

## Slide 2 — The Motivating Question

# Does your name affect how a search engine ranks you?

![Search engine processing two identical resumes with different names](presentation/01_name_swap.png)

> **Speaker Notes:**
>
> Let me start with a concrete scenario. Imagine two people apply for a job. Their resumes are **word-for-word identical** — same experience, same qualifications, same education. The only difference is the name at the top: one says "Emily," the other says "Lakisha."
>
> This isn't a hypothetical. In 2004, economists Bertrand and Mullainathan ran exactly this experiment with real employers and found that "Emily" got 50% more callbacks than "Lakisha."
>
> Now fast-forward to 2026. Many hiring platforms use AI-powered search engines to rank candidates. My question is: **does the same discrimination happen inside these algorithms?** And if so, can we figure out *why*?

## Slide 3 — The Retrieval Scenario

# What Exactly Are We Testing?

### The scenario: An employer searches for candidates

```
┌──────────────────────────────────┐     ┌─────────────────────────────────────┐
│          QUERY (Search)          │     │         DOCUMENTS (Candidates)       │
│                                  │     │                                     │
│ "Who is a qualified doctor?"     │────▶│ Doc A: "Emily has 10 years of       │
│                                  │     │         clinical experience."        │
│  (Employer's search input)       │     │                                     │
│                                  │────▶│ Doc B: "Lakisha has 10 years of     │
│                                  │     │         clinical experience."        │
└──────────────────────────────────┘     └─────────────────────────────────────┘
                                                       ↕ identical except name
```

- **Query** = employer's job search (stays the same)
- **Documents** = candidate descriptions (name gets swapped)
- **Score** = How relevant the AI thinks each candidate is

> **Speaker Notes:**
>
> Before diving into the technical details, let me be very precise about what we're testing.
>
> We're simulating a **candidate retrieval** scenario. Think of a hiring platform where an employer types "Who is a qualified doctor?" and the system returns ranked candidate profiles.
>
> The **query** — the search — stays the same. What we change is the **document** — the candidate's profile. We create two versions that are word-for-word identical except for one thing: the name. One says "Emily," the other says "Lakisha."
>
> Then we ask: does the AI search engine give them the same relevance score? If not, we can look inside and see exactly *which words* in the query responded differently.

---

## Slide 4 — How ColBERT Works

# ColBERT: The Only Model Where We Can Look Inside

### Step 1: Encode separately
```
Query:     "Who is a qualified doctor?"  →  [q₁, q₂, q₃, q₄, q₅, q₆]
Document:  "Emily has 10 years of..."   →  [d₁, d₂, d₃, d₄, d₅, ...]
```

### Step 2: Match each query word to its best document word
```
Score = best_match("Who")  +  best_match("is")  +  best_match("a")  
      + best_match("qualified")  +  best_match("doctor")  +  best_match("?")
```

### Why our experiment is valid

| Model | How It Scores | Affected by other docs? |
|---|---|---|
| **BM25** | Counts word matches | ❌ No |
| **DPR** | One embedding per doc | ❌ No |
| **ColBERT** | Per-word matching | ❌ **No** — each doc scored independently |

**Key property: The score for "Emily's" doc is exactly the same whether there are 2 documents or 8.8 million in the system.** Our pairwise comparison is mathematically identical to a full-scale search.

> **Speaker Notes:**
>
> ColBERT works in two steps. First, it **separately** encodes the query and the document into a sequence of numerical vectors — one per word. Then it computes a relevance score by matching each query word to its best-matching document word and summing up all the matches.
>
> This architecture has two properties that make it perfect for our study:
>
> **First**, because each word keeps its own vector, we can measure the score contribution of *every individual query word*. This is impossible in models like DPR that compress everything into a single number.
>
> **Second** — and this is crucial for understanding why our experiment is valid — ColBERT scores each document **independently**. The score it gives to Emily's profile doesn't depend on whether Lakisha's profile is also in the system, or whether there are a million other profiles. So when we compare just two documents side by side, we get *exactly the same scores* as we would in a production system with millions of candidates. Our simple pairwise experiment is mathematically equivalent to a full-scale retrieval.

---

## Slide 5 — Our Method: TCD

# Token Contribution Disparity (TCD)

![TCD mechanism showing per-word bias measurement](presentation/02_tcd_mechanism.png)

### The procedure:
1. **Same query** → "Who is a qualified doctor?"
2. **Two documents** → Only the name changes (Emily → Lakisha)
3. **Measure each query word's score change** → That's TCD

> **Speaker Notes:**
>
> So here's our method, which we call **Token Contribution Disparity**, or TCD.
>
> The idea is simple: take the same search query, score it against two identical candidate profiles that differ only by name, and measure how much **each query word's** individual score contribution changes.
>
> For the query "Who is a qualified doctor?" — we measure how much the word "Who" contributes differently, how much "is" shifts, how much "a" shifts, and so on. The name change happens in the *document*, but TCD measures the effect on the *query* side — because that's where ColBERT's per-word scoring lets us decompose the total score.
>
> If a query word's TCD is HIGH, that means identity information has "leaked" from the document into how the model matches that query word. If TCD is LOW, the word is unaffected by the name change.
>
> Now here's the key question: which words do you *expect* to show high TCD? Most people would guess "doctor" or "qualified" — the meaningful words. But that's not what we found.

---

## Slide 6 — The Surprising Finding

# Finding #1: Bias Hides in the Small Words

### Not here:
- ~~"doctor"~~ ← low TCD
- ~~"qualified"~~ ← low TCD
- ~~"experienced"~~ ← low TCD

### But HERE:
- **"is"** ← high TCD (72% biased toward majority group)
- **"a"** ← high TCD (64% biased)
- **"who"** ← high TCD (59% biased)
- **"?"** ← high TCD (69% biased)

### Function words carry **1.44×** more bias than content words

*p < 0.0001 across 43,065 controlled tests, 33 professions, 65 names*

> **Speaker Notes:**
>
> This is our most surprising and important finding.
>
> When we look at which words carry the most bias, it's NOT the words you'd expect. "Doctor," "qualified," "experienced" — the meaningful, content-rich words — are relatively stable when you swap names.
>
> Instead, the bias concentrates in the **small, seemingly meaningless words**: "is," "a," "who," the question mark.
>
> These are called **function words** — the grammatical glue of language. They have almost no meaning on their own. But in a neural network, precisely because they lack their own strong meaning, they **absorb the most influence from surrounding context** — including the person's name.
>
> On average, function words carry 1.44 times more bias than content words. This is statistically robust across 43,000 tests.

---

## Slide 7 — Why Does This Happen?

# The Mechanism: Why Function Words Absorb Bias

```
Step 1:  Name changes (Emily → Lakisha)
              │
Step 2:  AI re-reads the ENTIRE sentence
              │
         ┌────┴────────────────────────┐
         │                             │
    "experience"                  "is", "a", "has"
    Strong meaning anchor         No meaning anchor
    → barely shifts               → shifts A LOT
              │                        │
Step 3:  Query words match         Query "is" matches
         to similar targets        to DIFFERENT targets
              │                        │
Step 4:  Low TCD                   HIGH TCD → Score bias
```

### Measured embedding shift:
| | Function words | Content words |
|---|---|---|
| **How much the AI's internal representation changes** | Large shift (0.114) | Small shift (0.013) |
| **Ratio** |  | **8.9× difference** |

> **Speaker Notes:**
>
> Let me explain *why* this happens, in intuitive terms.
>
> When a neural network reads a sentence, every word influences every other word through a mechanism called "self-attention." Think of it as a conversation between all the words in a sentence.
>
> When you change "Emily" to "Lakisha," the AI re-processes the entire sentence. Words like "experience" and "clinical" have strong, stable meanings — they're like heavy anchors that don't move much regardless of context. But function words like "is" and "has" have *no intrinsic meaning* — they're like lightweight buoys that float wherever the current takes them.
>
> So when the name changes, the function words' internal representations shift dramatically — almost 9 times more than content words. And when these representations shift, they match to different parts of the query, causing the score to change.
>
> In other words: function words act as **sponges** that absorb bias from nearby identity words and transmit it into the final score.

---

## Slide 8 — Is This Specific to Identity?

# Control Experiment: Is This Just General Sensitivity?

Maybe ColBERT is just sensitive to *any* word change, not specifically names?

| What We Changed | Example | Bias Level |
|---|---|---|
| **Cross-gender pronoun** | He → They | **1.92×** baseline |
| **Cross-race name** | Emily → Lakisha | **1.24×** baseline |
| Same-race name | Emily → Jennifer | 1.14× baseline |
| Non-identity word | "ten years" → "fifteen years" | 1.00× (baseline) |

**Identity swaps cause significantly MORE function-word bias than non-identity swaps**

*p < 10⁻⁸, Cohen's d = 0.37*

> **Speaker Notes:**
>
> A reasonable objection is: "Maybe the model is just generally sensitive to word changes, and this has nothing specific to do with identity."
>
> So we ran a control experiment. We compared four types of substitutions:
> - Changing a pronoun across gender lines (He → They)
> - Changing a name across racial lines (Emily → Lakisha)
> - Changing a name within the same group (Emily → Jennifer)
> - Changing a completely non-identity word ("ten years" → "fifteen years")
>
> If the model were just generically sensitive, all four should show similar bias. But they don't. Identity-crossing swaps produce significantly more function-word bias than control swaps, and cross-gender swaps produce the strongest signal.
>
> This tells us: the effect is **specifically about identity**, not just about word-level noise.

---

## Slide 9 — Finding #2: The Tokenization Tax

# Finding #2: Your Name's "Complexity" Amplifies Bias

![BPE tokenization breaking names into subword pieces](presentation/03_tokenization_tax.png)

> **Speaker Notes:**
>
> Our second major finding is about how AI models process names at a fundamental level.
>
> Before a neural network can read your name, it has to break it into pieces called "tokens." Common names that appear frequently in the training data get their own single token. But rare names — which disproportionately belong to minority communities — get chopped up into fragments.
>
> "Emily" is one token. "Lakisha" becomes three tokens: "La," "ki," "sha." And a South Asian name like "Thiruvengadam" becomes five tokens.
>
> More fragments mean more uncertainty for the model. More uncertainty means the model relies more heavily on **patterns from its training data** — which contain societal biases.
>
> The result: rare names that get split into more tokens show **1.84 times** more bias than common single-token names. This is what we call the **Tokenization Tax** — an invisible fairness cost built into the very infrastructure of how AI processes language.

---

## Slide 10 — The Full Picture: 33 Professions

# This Pattern Holds Across 33 Professions

![Function-word vs content-word TCD across 33 professions](presentation/06_func_vs_cont.png)

*Function words > Content words in **30 out of 33** professions tested*

*Overall: Wilcoxon p = 2.2 × 10⁻⁴⁷*

> **Speaker Notes:**
>
> One concern is that this might only happen for certain professions. So we tested 33 different professions spanning healthcare, STEM, education, legal, leadership, blue-collar, and service sectors.
>
> The function-word bias pattern holds in 30 out of 33 professions. It's not driven by any single profession or category — it's a pervasive, systemic pattern.
>
> Professions with stronger gender stereotypes in society — like nurse, secretary, and plumber — tend to show slightly higher overall bias, but the *pattern* of function words carrying more bias than content words is consistent everywhere.

---

## Slide 11 — Decomposing Confounds

# Finding #3: What Are Name Swaps Really Measuring?

When we swap "Emily" → "Lakisha," we're changing **multiple things at once**:

| Factor | Emily | Lakisha |
|---|---|---|
| Race | White | Black |
| Gender | Female | Female |
| Name frequency | Very common (844/100K) | Very rare (5/100K) |
| BPE tokens | 1 | 3 |
| Socioeconomic signals | High SES | Low SES |

**Which factor actually drives the bias?**

We used **43,065 controlled tests** (all pairs of 30 names × 33 professions × 3 templates) with regression analysis to isolate each factor.

> **Speaker Notes:**
>
> Here's a subtlety that's really important for the broader implications of this work.
>
> When classic audit studies swap "Emily" for "Lakisha," they typically attribute the result to racial discrimination. But the name change actually alters *multiple things*: race, name frequency, tokenization complexity, and even socioeconomic signals.
>
> So we designed a large-scale decomposition experiment. We tested all possible pairings of 30 carefully selected names across 33 professions and 3 sentence templates — that's over 43,000 tests. Then we used regression analysis to untangle which factor independently drives the bias.

---

## Slide 12 — Decomposition Results

# The Decomposition Results

![Decomposition forest plot showing coefficients](presentation/08_decomposition.png)

### What independently predicts bias?

| Factor | Stable across models? | Significance |
|---|---|---|
| **Gender mismatch** | ✅ Always significant | p < 10⁻⁹ |
| **Tokenization mismatch** | ✅ Significant | p = 0.0005 |
| **Race mismatch** | ⚠️ Weakens with controls | p = 0.005 (weak) |
| Frequency gap | ❌ Not significant | p = 0.38 |

> **Speaker Notes:**
>
> The results challenge some common assumptions.
>
> **Gender mismatch** is the strongest, most stable predictor. Comparing a male name to a female name consistently produces more bias than same-gender comparisons, even after controlling for everything else. This is robust across every statistical specification we tried.
>
> **Tokenization mismatch** — how differently the AI breaks up two names — is the second strongest predictor. This is the tokenization tax I mentioned earlier, and it's independently significant.
>
> **Race mismatch** shows a weaker and less stable effect. In simple models, it appears significant. But as we add more control variables — frequency, tokenization, socioeconomic status — the race signal *weakens*. This doesn't mean race doesn't matter. It means that what previous work often attributed to "racial bias" may partly be driven by the confounded effects of name frequency and tokenization.
>
> This is a cautious but important finding: classic name-swap audits may be measuring a mixture of effects, not a pure racial signal.

---

## Slide 13 — Model Ladder

# How the Race Signal Fades with Better Controls

![Nested model ladder showing R² progression](presentation/10_model_ladder.png)

### Adding controls step by step:

```
Model 0: Race only           → Race p = 0.011 ✓
Model 1: + Gender            → Race p = 0.001 ✓, Gender p < 10⁻¹⁵ ✓
Model 2: + Tokenization      → Race p = 0.059 ⚠️ (weakening)
Model 3: + SES + Frequency   → Race p = 0.124 ✗ (no longer significant)
```

**Gender remains rock-solid across all specifications (p < 10⁻¹⁴)**

> **Speaker Notes:**
>
> This slide shows the nested model ladder — a technique from econometrics where you add control variables one at a time and watch what happens.
>
> When we model bias with *only* race as a predictor, it looks significant. But watch what happens as we add controls:
>
> Add gender — race stays significant but gender is *much* stronger.
> Add tokenization — race drops to marginal (p = 0.06).
> Add socioeconomic status and frequency — race is no longer statistically significant.
>
> Meanwhile, gender stays rock-solid at p < 10⁻¹⁴ no matter what else we add.
>
> This is an honest result. Many bias papers would stop at Model 0 and declare "racial bias confirmed." We dig deeper and find a more nuanced picture. For this model, gender and tokenization are the dominant bias channels.

---

## Slide 14 — Does This Hold in Real Text?

# Ecological Validity: Real Text Confirms the Pattern

![Cross-setting validation comparison](presentation/04_cross_setting.png)

| Setting | Func-TCD / Cont-TCD Ratio | p-value |
|---|---|---|
| Synthetic templates | 1.44× | 2.2 × 10⁻⁴⁷ |
| **Natural text passages** | **1.42×** | **6.4 × 10⁻²⁶** |
| BM25 (traditional search) | 0.00× | — |

Tested on **1,680** natural-text experiments across 7 text categories (news, medical, legal, academic, recommendation letters, etc.)

> **Speaker Notes:**
>
> A fair question is: does this hold outside of our controlled laboratory settings?
>
> We tested the function-word bias pattern on 12 realistic text passages — news articles, medical records, academic bios, legal testimonies, recommendation letters — spanning 7 categories. We ran 1,680 tests.
>
> The remarkable result: the function-word to content-word ratio goes from **1.44 times** in synthetic templates to **1.42 times** in natural text. That's almost exactly the same ratio. The absolute magnitudes decrease — real text has more linguistic diversity that dilutes the signal — but the *relative pattern* is preserved.
>
> And when we test BM25 — a traditional, non-AI search engine — the bias is literally **zero**. No function-word propagation at all. This confirms that the effect is specific to AI-based contextual models.

---

## Slide 15 — Why BM25 Shows Zero Bias

# The BM25 Control: Proof It's a Contextual Model Problem

### BM25 (Traditional Search)
```
"Emily has experience" → matches "experience" ← exact word match
"Lakisha has experience" → matches "experience" ← same exact match
```
Changing the name only affects the name token. No leakage.

### ColBERT (AI Search)
```
"Emily has experience" → AI reads whole sentence → "has" shifts
"Lakisha has experience" → AI reads whole sentence → "has" shifts differently
```
The name changes how the AI *understands every word*.

**BM25 Function-word TCD: exactly 0.000**

> **Speaker Notes:**
>
> Let me make this point really concrete. Why does BM25 show zero bias?
>
> BM25 works by counting exact word matches. When you search for "experienced doctor," it checks: does the document contain "experienced"? Does it contain "doctor"? It doesn't care about "Emily" or "Lakisha" because those words aren't in the query.
>
> ColBERT is fundamentally different. It reads the ENTIRE document and uses neural attention to build a contextual understanding of every word. When the name changes, the AI's understanding of "has," "is," and even "experience" subtly shifts — because in a neural network, everything is connected to everything.
>
> The BM25 result is our cleanest control: it proves that the function-word bias we observe is a product of contextual AI representations, not an artifact of our measurement technique.

---

## Slide 16 — The Bigger Picture

# What This Means

### 1. Bias is **distributed**, not localized
Traditional debiasing targets the "obvious" words (doctor → nurse bias). But bias actually spreads to every token through contextual representations.

### 2. Tokenization is an **invisible fairness tax**
Names that require more subword tokens — disproportionately minority names — face amplified bias. This is a form of **infrastructural inequality**.

### 3. Standard audit methods **oversimplify**
Classic Emily-vs-Lakisha audits measure a mixture of gender, tokenization, frequency, and race effects — not a pure racial signal.

### 4. ColBERT offers a **unique diagnostic capability**
Its word-by-word scoring lets us see what's invisible in other AI models.

> **Speaker Notes:**
>
> Let me step back and explain why these findings matter beyond computer science.
>
> First, bias in AI search is *distributed*, not localized. Most debiasing techniques try to remove associations between words like "doctor" and "male." But if the bias is actually flowing through "is" and "a," those techniques are aiming at the wrong target.
>
> Second, tokenization — the way AI breaks words into pieces — creates an invisible fairness tax. Your name gets chopped up more if it's rare, and rare names are disproportionately from minority communities. This is a form of what scholars call **infrastructural inequality** — it's built into the plumbing of the system, invisible to end users.
>
> Third, the classic name-swap audits that kicked off this whole field of research are measuring something more complicated than "pure racial discrimination." Gender and tokenization are stronger independent signals than race in this model.
>
> And finally, ColBERT's unique architecture gives us a diagnostic tool that simply doesn't exist for other AI models.

---

## Slide 17 — Implications for Practice

# Practical Implications

### For AI developers:
- ⚠️ **Debiasing the wrong tokens**: If you only debias content words ("doctor," "nurse"), you miss the main bias channel
- 🔧 **Potential fix**: Stabilize function-word embeddings via null-space projection (future work)

### For policymakers:
- 📋 Tokenizer design is a **fairness-relevant infrastructure choice**, not just a technical detail
- 🔍 Audit requirements should test for **distributed** bias, not just aggregate score differences

### For researchers:
- 🎯 TCD can be applied to **any** model that preserves per-token scores
- 📊 Name-swap audits should **always** control for tokenization and frequency confounds

> **Speaker Notes:**
>
> What should different stakeholders take away from this?
>
> For AI developers: you may be debiasing the wrong part of the model. If function words are the main bias channel, we need targeted interventions that stabilize function-word representations — not just adjust content-word associations.
>
> For policymakers: tokenizer design — how an AI breaks words into pieces — is currently treated as a purely technical choice. But our findings show it has direct fairness implications. Algorithmic audit requirements should be expanded to test for distributed, token-level bias, not just overall score differences.
>
> For other researchers: our TCD method is general. It can be applied to any search model that keeps per-word scores. And anyone doing name-swap audits should always control for tokenization and frequency confounds, or risk misattributing effects.

---

## Slide 18 — Experimental Scale

# By the Numbers

| | Scale |
|---|---|
| **Total controlled tests** | 48,000+ |
| **Names** | 65 (4 racial groups, 2 genders) |
| **Professions** | 33 (7 BLS categories) |
| **Templates** | 3–5 per experiment |
| **Experimental phases** | 7 (control, validation, expansion, decomposition, replication, real-text, BM25) |
| **Matched-pair families** | 5 (race, gender, tokenization, frequency, rarity) |
| **Statistical methods** | Wilcoxon, Mann-Whitney U, ANOVA, OLS with clustered SEs, fixed effects |
| **External data source** | Rosenman, Olivella & Imai (2023), Harvard Dataverse |

> **Speaker Notes:**
>
> Just to give you a sense of the scale: this isn't a small pilot study. We ran over 48,000 controlled counterfactual experiments across 65 names, 33 professions, multiple templates, and 7 distinct experimental phases.
>
> We used matched-pair designs borrowed from econometric audit studies to isolate each confounding factor. Our statistical approach includes cluster-robust standard errors and profession/template fixed effects to ensure rigorous inference.
>
> The name feature data is backed by a peer-reviewed external source — the Rosenman, Olivella & Imai dataset published in Scientific Data and hosted on Harvard Dataverse.

---

## Slide 19 — Limitations & Future Work

# Limitations and Next Steps

### Current Limitations
- 📌 **Single model**: Only ColBERTv2 tested (DPR, SPLADE comparisons planned)
- 📌 **Mostly synthetic text**: Template-based data (natural text validated but limited)
- 📌 **US-centric names**: 4 racial categories, does not cover all global contexts
- 📌 **No downstream ranking test**: 3% SS → does it change search results? (MS MARCO evaluation planned)

### Planned Next Steps
1. **MS MARCO benchmark evaluation** → Test on real search queries
2. **Multi-model comparison** → DPR & SPLADE
3. **Debiasing proof-of-concept** → Null-space projection on function-word embeddings
4. **Target venue: EMNLP 2026**

> **Speaker Notes:**
>
> I want to be transparent about what this study does NOT yet show.
>
> We've only tested one AI model. While ColBERT is important and widely used, we need to verify whether the same patterns appear in other models like DPR and SPLADE. That work is planned.
>
> Most of our experiments use synthetic template sentences, though we did validate the pattern on natural text passages.
>
> Our name pool covers four racial categories in the US context. Different cultural contexts may produce different patterns.
>
> And perhaps most importantly, we haven't yet shown that our measured 3% score sensitivity actually changes search results in practice. A full evaluation on standard retrieval benchmarks like MS MARCO is the natural next step.
>
> We're targeting EMNLP 2026 for publication.

---

## Slide 20 — Thank You

# Thank You

## Key Takeaway

> **Bias in AI search doesn't hide where you expect.**
>
> It doesn't live in "doctor" or "nurse." It lives in "is," "a," and "who" — the invisible grammatical glue of language. And it's amplified by the very way AI processes unfamiliar names.

### Questions?

📧 ziyu_kong@brown.edu

📄 Paper & Code: github.com/[repo]

> **Speaker Notes:**
>
> To summarize: the single most important takeaway from this research is that bias in AI search is **distributed and hidden** in the places you'd least expect.
>
> We think of bias as living in stereotypical associations — "doctor" with "male," "nurse" with "female." But our X-ray of ColBERT reveals that bias actually flows through the smallest, most invisible words in a sentence: "is," "a," "who."
>
> And it's amplified by an infrastructural design choice — tokenization — that disproportionately affects names from minority communities.
>
> This matters because if you're trying to fix bias, you need to know where it actually is. And now, for the first time, we have a tool that can show you.
>
> Thank you. I'm happy to take questions.
