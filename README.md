# ColBERT Bias Audit: Token-Level Attribution in Dense Retrieval

## Overview

This repository contains the code, data, and experiments for analyzing identity-related bias in ColBERT (late-interaction dense retrieval) using **Token Contribution Disparity (TCD)** — a decomposable metric that attributes bias to individual query tokens.

## Key Findings

1. **Function-word sensitivity**: Function words (`who`, `is`, `an`, punctuation) absorb 1.43× more identity-induced score change than content words across 55,440 controlled ColBERT tests.
2. **Template and passage effects**: The function/content ratio varies by document template (0.96×–1.84×) and rises to 2.08× in a naturalistic-template validation set.
3. **Hierarchical robustness**: The core function/content ratio remains above 1.0 under name-pair, profession, template, and name-pair×profession cluster bootstraps; mixed-effects models with repeated design factors also retain a positive function-word effect.
4. **Mechanistic check**: MaxSim argmax tracing and an aggregate transition heatmap rule out a simple “function words jump to the name token” explanation; function words have higher TCD without higher argmax-switch or name-context-hit rates.
5. **Confound decomposition**: Gender mismatch is the most stable observed predictor of absolute score sensitivity; cross-race pairing does not independently amplify absolute sensitivity after controls.
6. **Rarity amplification**: Rare–rare name pairs show a modest score-sensitivity increase over common–common pairs (RR/CC = 1.11×).

## Repository Structure

```
├── paper/                  # Publication manuscript (EMNLP target)
├── src/                    # Core library code (TCD, models, utilities)
├── experiments/            # Numbered experiment scripts (reproducible)
│   ├── 01_control_validation/
│   ├── 02_function_word_bias/
│   ├── 03_profession_expansion/
│   ├── 04_name_confound/
│   ├── 05_real_text_validation/
│   ├── 06_bm25_baseline/
│   ├── 07_multi_model/
│   ├── 08_msmarco_counterfactual/
│   ├── 09_ranking_impact/
│   ├── 10_embedding_probing/
│   └── 10_mitigation/
├── data/                   # Name pools, professions, passages
├── results/                # Experiment outputs (numbered)
├── scripts/                # Reproducibility scripts
├── configs/                # Experiment configurations
└── docs/                   # Course materials, presentations, archives
```

## Reproduction

```bash
# Install dependencies
pip install -r requirements.txt

# Run all experiments
bash scripts/run_all.sh

# Generate paper figures
python local/regenerate_all_figures.py

# Robustness and mechanism checks
python experiments/30_scaled_experiment/cluster_bootstrap.py
python experiments/30_scaled_experiment/mixed_effects.py
python experiments/14_argmax_alignment/plot_argmax_heatmap.py
```

## Citation

```bibtex
@misc{anonymous2026tcd,
  title={Where Does Bias Hide in Neural Retrieval? Token-Level Attribution in Late-Interaction Models},
  author={Anonymous},
  year={2026},
  note={Anonymous submission}
}
```

## License

MIT
