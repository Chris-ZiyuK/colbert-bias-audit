# Finding: Bias in ColBERT Propagates Through Function Words, Not Content Words

**CSCI 2952W — Critical AI and Data Studies**  
**Date**: March 8, 2026

---

## Summary

Through a series of controlled experiments on the ColBERTv2 dense retrieval model, we discovered an unexpected and, to our knowledge, previously unreported phenomenon: **when identity markers (names, pronouns) are swapped in otherwise identical documents, the resulting bias in retrieval scores is primarily carried by function words (e.g., "is", "who", "a"), not by the semantically meaningful content words (e.g., "doctor", "nurse", "qualified").**

This finding challenges the intuitive assumption that bias in retrieval systems operates through direct lexical associations (e.g., "nurse" ↔ "female"). Instead, we show that bias in ColBERT is a **distributed, context-mediated phenomenon** that propagates through self-attention and manifests most strongly in tokens with weak intrinsic semantics.

---

## Experimental Evidence

### Scale

We tested **650 counterfactual pairs** spanning 10 professions, 13 identity swaps (10 name-based across 4 racial groups + 3 pronoun-based across gender categories), and 5 document templates. For each pair, we decomposed the ColBERT MaxSim score into **per-query-token contributions** and measured the Token Contribution Disparity (TCD) — the change in each query token's contribution when the identity marker is swapped.

### Result 1: Function Words Have Higher TCD

We classified all query tokens into **function words** (is, who, a, an, ?) and **content words** (doctor, nurse, qualified, talented, etc.) and compared their |TCD| distributions:

|  | N | Mean |TCD| | Median |
|---|---|---|---|
| **Function words** | 2,535 | **0.0210** | 0.0133 |
| Content words | 4,225 | 0.0146 | 0.0088 |

**Mann-Whitney U test** (function > content): *p* = 0.0002 ***  
**Ratio**: Function word bias is **1.44×** that of content words.

### Result 2: Consistency Across Professions

This pattern holds in **9 out of 10 professions** tested:

| Profession | Function/Content Ratio |
|---|:---:|
| experienced doctor | 2.24× |
| talented researcher | 1.85× |
| competent lawyer | 1.84× |
| reliable nurse | 1.68× |
| qualified software engineer | 1.54× |
| brilliant scientist | 1.52× |
| successful business leader | 1.49× |
| dedicated teacher | 1.46× |
| caring social worker | 1.03× |
| skilled electrician | 0.89× ← only exception |

### Result 3: Directionality

Among function words, bias is not random — it shows consistent directionality:
- "is": 72.3% of the time favors the majority/male group (*p* < 0.0001 ***)
- "?": 68.6% favors majority (*p* < 0.0001 ***)
- "a": 64.3% favors majority (*p* < 0.0001 ***)
- "who": 59.1% favors majority (*p* = 0.00001 ***)

---

## Mechanistic Explanation

### Embedding Perturbation Analysis

To understand *why* function words carry more bias, we directly measured how much each token's 128-dimensional embedding vector changes when the identity marker is swapped (e.g., "Emily" → "Lakisha"). We computed the cosine distance (1 − cosine similarity) between the embeddings of the same token before and after the swap:

| Token Type | Mean Embedding Shift | N |
|---|:---:|:---:|
| **Function words** | **0.1137** | 9 |
| Content words | 0.0128 | 6 |

**Function word embeddings shift 8.9× more** than content word embeddings (*p* = 0.0002 ***).

### Why This Happens

The explanation lies in how BERT (the backbone of ColBERT) constructs contextualized embeddings:

1. **BERT uses self-attention**, meaning every token's embedding is influenced by all other tokens in the sentence. When we change one token (the identity marker), self-attention re-computes embeddings for *every* token.

2. **Content words have strong intrinsic semantics.** A word like "clinical" or "experience" has a robust dictionary meaning that anchors its embedding. Even when the context changes, its vector remains relatively stable — self-attention adjusts it only slightly.

3. **Function words have weak intrinsic semantics.** Words like "has", "is", and "a" carry almost no meaning on their own — their embeddings are almost entirely determined by context. When the context changes (identity swap), their vectors shift dramatically because there is no semantic anchor to hold them in place.

4. **ColBERT's MaxSim amplifies this effect.** In MaxSim scoring, each query token searches for its best-matching document token. Because function word embeddings in the document are the most volatile, the MaxSim match scores for query function words change the most after an identity swap — producing higher TCD.

In diagram form:

```
Identity swap (Emily → Lakisha)
       │
       ▼
Self-attention re-computes ALL document token embeddings
       │
       ├── Content words ("clinical", "experience"): small shift (anchored by meaning)
       │
       └── Function words ("has", "is", "a"): LARGE shift (no semantic anchor)
                │
                ▼
       Query function words ("who", "is", "a") find different MaxSim matches
                │
                ▼
       Score changes → measured as TCD → aggregates to overall SS bias
```

---

## Implications

### For Bias Research

This finding suggests that **bias in contextual retrieval models is fundamentally different from bias in static word embeddings.** In static models (Word2Vec, GloVe), bias operates through direct lexical associations — the vector for "nurse" sits closer to "woman." In contextual models like ColBERT, bias is a distributed phenomenon that "leaks" through the self-attention mechanism into tokens that have no direct semantic relationship with the identity being tested.

### For Debiasing

Traditional debiasing approaches (e.g., Bolukbasi et al., 2016) target specific word pairs or learned gender subspaces. Our finding implies that **these approaches may be insufficient for contextual retrieval models**, because the bias is not localized in the identity token's embedding — it spreads across all context-dependent tokens. Effective debiasing for ColBERT may require intervention at the attention layer or a context-aware null-space projection that accounts for the distributed nature of the bias.

### For ColBERT's Architecture

ColBERT's late interaction mechanism (MaxSim) makes this distributed bias *measurable* at the token level — something that single-vector models (DPR, ANCE) cannot do because they collapse all tokens into one embedding. This positions ColBERT not just as a retrieval model, but as a **diagnostic instrument** for understanding contextual bias in dense retrieval systems.

---

## Limitations

1. **Synthetic data**: All experiments use template-based counterfactual pairs. Validation on natural text (e.g., MS MARCO) is needed.
2. **Small embedding perturbation sample**: The mechanistic analysis uses only 3 document pairs (15 token observations). A larger study is planned.
3. **Single model**: Only ColBERTv2 was tested. Cross-model comparison is needed.
4. **Effect size**: While statistically significant, the absolute difference in mean |TCD| between function and content words (~0.006) is modest. Its practical impact on ranking needs further investigation.
