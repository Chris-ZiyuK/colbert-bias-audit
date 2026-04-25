#!/usr/bin/env bash
# ===========================================================================
# run_all.sh — Reproduce all experiments from scratch
# ===========================================================================
# Usage:  bash scripts/run_all.sh
# Requires: conda environment 'colbert_bias' with dependencies installed
# ===========================================================================

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

echo "═══════════════════════════════════════════════════════════"
echo "  ColBERT Bias Audit — Full Reproduction Pipeline"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Phase 1: Control validation
echo "▶ [01] Control Validation..."
python experiments/01_control_validation/control_test.py

# Phase 2: Function-word bias (initial 10 professions)
echo "▶ [02] Function-Word Bias..."
python experiments/02_function_word_bias/function_word_bias.py

# Phase 3: Profession expansion (33 professions)
echo "▶ [03] Profession Expansion..."
python experiments/03_profession_expansion/profession_expansion.py
python experiments/03_profession_expansion/p1_wrapup_analysis.py

# Phase 4: Name confound decomposition
echo "▶ [04] Name Confound Decomposition..."
python experiments/04_name_confound/p2_name_confound_audit.py
python experiments/04_name_confound/p2b_rosenman_validation.py

# Phase 5: Real-text validation
echo "▶ [05] Real-Text Validation..."
python experiments/05_real_text_validation/real_text_validation.py

# Phase 6: BM25 baseline
echo "▶ [06] BM25 Baseline..."
python experiments/06_bm25_baseline/bm25_baseline.py

# Phase 7: DPR comparison (planned)
# echo "▶ [07] DPR Comparison..."
# python experiments/07_dpr_comparison/dpr_comparison.py

# Phase 8: MS MARCO counterfactual (planned)
# echo "▶ [08] MS MARCO Counterfactual..."
# python experiments/08_msmarco_counterfactual/msmarco_counterfactual.py

# Phase 9: Ranking impact (planned)
# echo "▶ [09] Ranking Impact..."
# python experiments/09_ranking_impact/ranking_impact.py

# Phase 10: Mitigation (planned)
# echo "▶ [10] Mitigation..."
# python experiments/10_mitigation/tokenizer_equalization.py

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✅ All experiments complete."
echo "═══════════════════════════════════════════════════════════"
