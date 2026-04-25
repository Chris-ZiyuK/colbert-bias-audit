# Where Does Bias Hide? Token-Level Attribution Reveals Distributed Bias Propagation in Late-Interaction Retrieval

**Chris Ziyu Kong**  
Brown University · `ziyukong@brown.edu`

---

## Abstract

We present a mechanism-level audit of identity-related bias in ColBERT, a late-interaction dense retrieval model. Using ColBERT's decomposable MaxSim scoring, we introduce *Token Contribution Disparity* (TCD), a metric that attributes bias to individual query tokens after counterfactual identity swaps. Across 43,065 controlled tests spanning 33 professions and 65 names, we find that: (1) bias propagates primarily through function words ("is," "who," "a"), not through semantically relevant content words, at a ratio of **1.44×**; (2) rare names—those fragmented into more subword tokens by BPE—amplify this effect by up to **1.84×**; (3) the strongest confound-decomposition signals are cross-gender and tokenization mismatch, with a weaker residual race-category effect; (4) the function-word propagation pattern holds at nearly identical magnitude (**1.42×**, p < 10⁻²⁵) in naturalistic prose; and (5) a BM25 control produces *zero* function-word propagation, confirming this is a contextual-representation phenomenon. Together, these findings reveal that bias in transformer-based retrieval is distributed and context-mediated—not localized in lexical associations—with implications for both measurement and mitigation.

---

## 1. Introduction

Search systems shape information access. When a retrieval model ranks documents about a person, the ranking implicitly distributes attention, visibility, and opportunity (Singh & Joachims, 2018). If identity cues—names or pronouns—affect relevance scores, the system participates in discriminatory allocation even when no explicit demographic filter is applied.

Prior work has demonstrated that neural retrieval models produce measurably biased outputs (Rekabsaz et al., 2021; Ziems et al., 2024), and that names alone can distort embedding similarity (Manchanda & Shivaswamy, 2025). However, these studies operate at the *aggregate* level: they show *that* systems are unfair without explaining the internal mechanism through which identity information changes relevance scores.

We pursue a different question: **Where inside a neural retriever does identity-related bias actually reside, and how does it propagate to the output score?**

ColBERT (Khattab & Zaharia, 2020; Santhanam et al., 2022) provides a unique opportunity to answer this question. Unlike single-vector models such as DPR (Karpukhin et al., 2020), ColBERT's *late-interaction* architecture preserves token-level document representations and computes relevance through fine-grained MaxSim matching. This means the total score can be decomposed into individual query-token contributions. By swapping identity markers in otherwise identical documents and observing which query tokens' contributions change, we can localize the pathways through which bias propagates.

### Contributions

1. **TCD metric.** We introduce Token Contribution Disparity, the first token-level bias attribution metric for dense retrieval (§2).
2. **Function-word bias propagation.** We show that bias flows through semantically vacuous function words ("is," "who," "a"), not through content words ("doctor," "nurse"), overturning the intuition that bias equals lexical association (§4.1).
3. **Rarity amplification.** Rare names that fragment into more BPE tokens amplify bias by up to 1.84×, revealing tokenization as a hidden fairness tax (§4.2).
4. **Confound decomposition.** A structured name-pair design isolates gender, tokenization, frequency, and race as bias drivers, finding composite rather than single-factor explanations (§4.3).
5. **Ecological validation.** The core pattern generalizes from synthetic templates to naturalistic prose and is absent in BM25, confirming it is specific to contextual models (§4.4).

---

## 2. Related Work

### Fairness in retrieval
Singh & Joachims (2018) formalized exposure fairness, showing that ranking systems distribute attention unevenly. Rekabsaz et al. (2021) measured gender bias in BERT-based rankers and proposed adversarial mitigation, but their analysis remains at the aggregate-ranking level. Krieg et al. (2023) created the Grep-BiasIR dataset for gender representation bias. Our work differs by tracing bias to its *internal* mechanism rather than its statistical summary.

### Names as identity proxies
Bertrand & Mullainathan (2004) established name-swap audit methodology in their landmark resume experiment. Sweeney (2013) showed similar effects in online advertising. Caliskan et al. (2017) demonstrated with WEAT that static embeddings absorb human-like biases, and De-Arteaga et al. (2019) showed stereotyped representations persist even when explicit gender markers are removed. Manchanda & Shivaswamy (2025) found that names distort embedding similarity in isolation. We adopt the name-swap framework but go beyond aggregate scores to analyze *token-level* mechanisms.

### Tokenization and representation
Bostrom & Durrett (2020) showed that subword tokenization choices affect downstream model behavior. Sennrich et al. (2016) introduced BPE for neural MT, which forms the basis of most modern tokenizers. For retrieval, Goldfarb-Tarrant et al. (2024) showed that identity-linked information remains encoded in dense retrieval representations. We integrate tokenization into the bias story by demonstrating that BPE fragmentation directly mediates bias magnitude.

### ColBERT and late interaction
ColBERT (Khattab & Zaharia, 2020) and ColBERTv2 (Santhanam et al., 2022) are the only widely used retrievers that preserve per-token representations and compute scores via MaxSim. This makes them uniquely suited for token-level attribution, a property that single-vector models (Karpukhin et al., 2020) fundamentally lack.

---

## 3. Methodology

### 3.1 Setup and Definitions

We study ColBERTv2 (Santhanam et al., 2022), which encodes queries and documents into per-token embeddings and computes relevance as:

$$S(Q, D) = \sum_{i=1}^{|Q|} \max_{j \in [1,|D|]} \mathbf{q}_i^\top \mathbf{d}_j$$

where **q**ᵢ and **d**ⱼ are L2-normalized embeddings of the i-th query token and j-th document token.

A **counterfactual pair** (D_A, D_B) consists of two documents identical except for an identity marker (name or pronoun).

### 3.2 Token Contribution Disparity (TCD)

We define the **Token Contribution Disparity** for query token qᵢ as:

$$\text{TCD}(q_i) = |c_i^A - c_i^B|$$

where c_i^A and c_i^B are the MaxSim contributions from qᵢ to documents A and B. We then compute:

- **Func-TCD**: Mean |TCD| over function-word query tokens
- **Cont-TCD**: Mean |TCD| over content-word query tokens
- **Score Sensitivity (SS)**: Normalized total score difference

### 3.3 Name Pool and Confound Decomposition

Our experiments use 65 names from the Bertrand & Mullainathan (2004) audit set expanded via Rosenman et al. (2023). Each name is annotated with race, gender, BPE token count, and within-race frequency.

Five matched-pair families isolate individual confounds:

| Family | Match on | Vary |
|--------|----------|------|
| Race | gender, tokens, frequency | race |
| Gender | race, tokens, frequency | gender |
| Tokenization | race, gender, frequency | BPE token count |
| Frequency gap | race, gender, tokens | log-frequency gap |
| Absolute rarity | race, gender, tokens, frequency | mean rarity (median split) |

### 3.4 Experimental Phases

| Phase | Tests | Description |
|-------|------:|-------------|
| P0 | 1,080 | Control: identity vs. non-identity swaps |
| P1 | 1,287 | 33 professions × 3 swap types |
| P2/P2b | 43,065 | Name confound decomposition |
| RT | 1,680 | Real-text ecological validation |
| BM25 | 1,400 | Non-contextual baseline control |

---

## 4. Results

### 4.1 Function Words Carry More Bias Than Content Words

Across 1,287 P1 tests spanning 33 professions:

| Token Type | Mean |TCD| | Ratio | p-value |
|------------|:-----------:|:-----:|:-------:|
| **Function** | **0.0210** | **1.44×** | 0.0002 |
| Content | 0.0146 | — | — |

- Pattern holds in **30 of 33** professions (Wilcoxon p = 2.2 × 10⁻⁴⁷)
- Pronoun swaps produce stronger effects than name swaps (p = 0.003)
- Function-word embeddings shift **8.9×** more than content-word embeddings after identity swap

**Mechanism**: BERT's self-attention recomputes all token embeddings when any input changes. Content words have strong semantic anchors; function words are almost entirely context-determined. ColBERT's MaxSim amplifies this by making per-token matching scores the basis of the final ranking.

### 4.2 Rarity Amplification

| Name-pair rarity | Mean SS | Mean Func-TCD | Ratio |
|:-----------------|:-------:|:-------------:|:-----:|
| Common–Common | 0.0210 | 0.0108 | 1.0× |
| Common–Rare | 0.0305 | 0.0163 | 1.5× |
| **Rare–Rare** | **0.0366** | **0.0199** | **1.84×** |

ANOVA: Profession (F=214, p<0.0001) and Rarity (F=300, p<0.0001) are both highly significant. BPE tokenization functions as a *hidden fairness tax* on minority-group names.

### 4.3 Confound Decomposition

P2b Rosenman-backed targeted validation (65 names, OLS with pair-clustered SEs):

| Factor | Pairs | SS coef | Func-TCD coef | Func-TCD p |
|--------|:-----:|:-------:|:-------------:|:----------:|
| **Gender** | 107 | +0.0088 | **+0.0049** | **4.5 × 10⁻⁹** |
| **Tokenization** | 103 | +0.0067 | **+0.0048** | **5.3 × 10⁻⁴** |
| **Race** | 124 | +0.0043 | +0.0026 | **4.9 × 10⁻³** |
| Frequency gap | 100 | +0.0015 | +0.0008 | 0.383 |
| Absolute rarity | 28 | +0.0024 | +0.0032 | 0.084 |

**Key interpretation**: Name-swap bias is a **composite effect** — not reducible to race alone. Cross-gender contrast and tokenization mismatch are the strongest, most stable drivers.

### 4.4 Ecological Validation and Cross-Model Comparison

#### Real-text generalization

12 naturalistic passages × 10 queries × 14 name pairs = 1,680 tests:

| Setting | Func-TCD | Cont-TCD | Ratio | p |
|---------|:--------:|:--------:|:-----:|:-:|
| Synthetic templates | 0.0177 | 0.0133 | 1.44× | 2.2 × 10⁻⁴⁷ |
| **Natural text** | **0.0124** | **0.0087** | **1.42×** | **6.4 × 10⁻²⁶** |
| BM25 baseline | 0.0000 | 0.0000 | — | — |

The function-word dominance pattern is remarkably consistent: **1.42×** in natural text vs. **1.44×** in synthetic templates. The pattern holds across all 7 text categories.

#### BM25 control

BM25 (Robertson & Zaragoza, 2009) uses exact lexical matching. Name swaps produce **zero** function-word TCD — only name tokens are affected. This confirms that the ColBERT effect is a consequence of **contextual representations**, not of retrieval decomposition itself.

---

## 5. Discussion

### Bias as Distributed Phenomenon

Our central finding challenges the dominant framing of bias in NLP systems. Traditional bias narratives assume lexical association: "nurse" ↔ "female" in the embedding space (Bolukbasi et al., 2016; Caliskan et al., 2017). In contextual retrieval models, bias is instead *distributed*: it propagates through self-attention into tokens that have no semantic relationship with the identity being tested.

### Tokenization as Infrastructure-Level Bias

BPE tokenization is a *hidden fairness tax*: names common in majority populations ("Emily," "Greg") survive as single tokens, while minority names ("Lakisha," "Thiruvengadam") are fragmented. Fragmentation increases model uncertainty and amplifies reliance on stereotyped contextual priors.

### ColBERT as Diagnostic Instrument

ColBERT is not just a retrieval model — it is a **diagnostic instrument** for contextual bias. Single-vector models may exhibit the same or worse bias, but because they collapse all tokens into one embedding, there is no way to ask *which token caused the score difference*.

### Implications for Debiasing

Our finding that bias concentrates in function words suggests *function-word-specific null-space projection*: removing identity information from function-word representations while leaving content words intact. This approach is theoretically motivated but requires empirical validation.

---

## 6. Limitations

1. **Synthetic and semi-synthetic data.** Primary analyses use template-generated pairs; full MS MARCO evaluation needed.
2. **Single model.** Only ColBERTv2 tested; comparison with DPR/SPLADE needed.
3. **Name pool scope.** 65 names covering four US-centric racial categories.
4. **Absolute rarity.** Marginal significance (p=0.084) with 28 pairs.
5. **No debiasing evaluation.** Mechanism identified but mitigation not implemented.

---

## 7. Conclusion

We have shown that identity-related bias in ColBERT does not reside where intuition suggests. It is not the word "doctor" or "nurse" that carries bias into the retrieval score — it is the word "is." This distributed, context-mediated propagation through function words is (1) robust across 33 professions, (2) amplified by BPE tokenization of rare names, (3) driven primarily by gender contrast and tokenization mismatch rather than race alone, (4) replicated at near-identical magnitude in naturalistic text, and (5) entirely absent in non-contextual retrieval. These findings reframe both the measurement and the mitigation of bias in neural retrieval systems: bias is not a localized pathology but a systemic property of contextual representations.

---

## Acknowledgments

This work was conducted as part of CSCI 2952W at Brown University. Rosenman et al. first-name ethnicity data from Harvard Dataverse (DOI: 10.7910/DVN/SGKW0K).

---

## References

- Bertrand, M. & Mullainathan, S. (2004). Are Emily and Greg More Employable than Lakisha and Jamal? *American Economic Review*, 94(4), 991–1013.
- Blodgett, S. L. et al. (2020). Language (Technology) is Power. *ACL 2020*, 5454–5476.
- Bolukbasi, T. et al. (2016). Man is to Computer Programmer as Woman is to Homemaker? *NeurIPS 2016*.
- Bostrom, K. & Durrett, G. (2020). BPE is Suboptimal for LM Pretraining. *Findings of EMNLP 2020*.
- Caliskan, A. et al. (2017). Semantics Derived Automatically from Language Corpora Contain Human-like Biases. *Science*, 356(6334), 183–186.
- De-Arteaga, M. et al. (2019). Bias in Bios. *FAccT 2019*, 120–128.
- Devlin, J. et al. (2019). BERT. *NAACL 2019*, 4171–4186.
- Goldfarb-Tarrant, S. et al. (2024). MultiContrievers. *BlackboxNLP 2024*, 87–100.
- Karpukhin, V. et al. (2020). Dense Passage Retrieval. *EMNLP 2020*, 6769–6781.
- Khattab, O. & Zaharia, M. (2020). ColBERT. *SIGIR 2020*, 39–48.
- Krieg, K. et al. (2023). Grep-BiasIR. *CHIIR 2023*.
- Manchanda, S. & Shivaswamy, P. (2025). What is in a Name? *Findings of ACL 2025*.
- Rekabsaz, N. et al. (2021). Societal Biases in Retrieved Contents. *SIGIR 2021*, 306–316.
- Robertson, S. & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *FnTIR*, 3(4), 333–389.
- Rosenman, E. T. R. et al. (2023). Race and Ethnicity Data for First, Middle, and Surnames. *Scientific Data*, 10, 299.
- Santhanam, K. et al. (2022). ColBERTv2. *NAACL 2022*, 3715–3734.
- Sennrich, R. et al. (2016). Neural MT of Rare Words with Subword Units. *ACL 2016*, 1715–1725.
- Singh, A. & Joachims, T. (2018). Fairness of Exposure in Rankings. *KDD 2018*, 2219–2228.
- Sweeney, L. (2013). Discrimination in Online Ad Delivery. *CACM*, 56(5), 44–54.
- Ziems, C. et al. (2024). Measuring and Addressing Indexical Bias in IR. *Findings of ACL 2024*.
