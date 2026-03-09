# ColBERT Bias Audit: Token-Level Attribution of Identity Bias in Dense Retrieval

> **Where does bias hide?** We show that identity-related bias in ColBERT propagates not through semantic
> content words (doctor, nurse), but through **function words** (is, who, a) — revealing a distributed,
> context-mediated bias mechanism that traditional debiasing methods cannot address.

## Key Findings (Preliminary)

| Finding | Evidence | Status |
|---------|----------|--------|
| **Function-Word Bias Leakage**: Function words carry 1.4× more bias (TCD) than content words | 650 pairs, p=0.0002 | ✅ Validated |
| **Identity Specificity**: The effect is stronger for identity swaps than ordinary swaps | P0 control, p<1e-8, d=0.37 | ✅ Validated |
| **Rarity Amplification**: Rare names (more BPE tokens) amplify bias by 1.84× | 1,140 tests, ANOVA p<0.0001 | ✅ Validated |
| **Real-text generalization**: Effect persists in natural (non-template) text | — | 🔲 Pending |
| **Multi-model contrast**: ColBERT vs DPR/BM25 comparison | — | 🔲 Pending |

## Project Structure

```
colbert-bias-audit/
├── src/
│   ├── models/          # Model wrappers (ColBERT, DPR, SPLADE)
│   ├── metrics/         # TCD, SS, Gini coefficient, F/C ratio
│   ├── audit/           # Core token-level audit logic
│   └── visualization/   # Heatmaps, TCD bar charts, interaction plots
├── data/
│   ├── counterfactual/  # Counterfactual pair datasets
│   └── audit_names/     # Name lists (Bertrand & Mullainathan, 2004)
├── experiments/
│   ├── phase0_control/  # P0: Identity vs ordinary swap control
│   ├── phase1_synthetic/# Phase 1: Synthetic template experiments
│   ├── phase2_real_text/ # Phase 2: MS MARCO natural text validation
│   └── phase3_multi_model/ # Phase 3: Cross-model comparison
├── results/             # Experiment outputs (JSON, CSV, PNG)
├── paper/               # LaTeX manuscript
├── configs/             # Experiment configurations (YAML)
├── notebooks/           # Jupyter analysis notebooks
├── project_overview.html # 项目讲解 (中文)
└── progress_tracker.html # 进度看板
```

## Quick Start

```bash
# Environment setup
conda create -n colbert_bias python=3.10
conda activate colbert_bias
pip install torch transformers matplotlib seaborn scipy statsmodels pandas

# Run the P0 control experiment
python experiments/phase0_control/p0_control_experiment.py
```

## Core Methodology: TCD (Token Contribution Disparity)

ColBERT computes relevance via **MaxSim**: for each query token, find the best-matching
document token and sum the scores. This decomposability lets us define:

```
TCD(q_i) = MaxSim(q_i, D_A) − MaxSim(q_i, D_B)
```

where D_A and D_B are counterfactual documents differing only in identity markers.
TCD tells us **exactly which query word** is responsible for the score difference.

## Citation

```bibtex
@article{kong2026colbert-bias,
  title={Where Does Bias Hide? Token-Level Attribution Reveals Distributed 
         Bias Propagation in Late-Interaction Retrieval},
  author={Kong, Chris Ziyu},
  year={2026}
}
```

## License

MIT
