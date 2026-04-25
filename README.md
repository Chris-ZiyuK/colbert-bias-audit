# ColBERT Bias Audit: Token-Level Attribution Reveals Distributed Bias in Dense Retrieval

## Overview

This repository contains the code, data, and experiments for analyzing identity-related bias in ColBERT (late-interaction dense retrieval) using **Token Contribution Disparity (TCD)** — a decomposable metric that attributes bias to individual query tokens.

## Key Findings

1. **Function-word bias propagation**: Bias flows through function words (`is`, `who`, `a`) at 1.44× the rate of content words, across 30/33 professions.
2. **Rarity amplification**: Names fragmented into more BPE subword tokens amplify bias by up to 1.84×.
3. **Confound decomposition**: Gender mismatch and tokenization disparity are stronger bias drivers than race category alone.

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
│   ├── 07_dpr_comparison/       # planned
│   ├── 08_msmarco_counterfactual/ # planned
│   ├── 09_ranking_impact/       # planned
│   └── 10_mitigation/          # planned
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
python scripts/generate_figures.py
```

## Citation

```bibtex
@inproceedings{kong2027tokenization,
  title={The Tokenization Tax: How BPE Fragmentation Amplifies Identity Bias in Dense Retrieval},
  author={Kong, Chris Ziyu},
  year={2027},
  booktitle={Proceedings of EMNLP 2027}
}
```

## License

MIT
