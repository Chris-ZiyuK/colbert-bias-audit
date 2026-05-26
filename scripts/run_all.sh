#!/usr/bin/env bash
# =============================================================================
# ColBERT Bias Audit — Full Experiment Pipeline
# =============================================================================
# Usage:  bash scripts/run_all.sh [PHASE]
#   PHASE: p0|p1|p2|p2b|rt|bm25|ablation|multi|msmarco|ranking|mitigation|all
#
# Prerequisites:
#   pip install -r requirements.txt
#   python -m spacy download en_core_web_sm  (for MS MARCO experiment)
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

PHASE="${1:-all}"
RESULTS="results"

echo "======================================================================"
echo "ColBERT Bias Audit Pipeline"
echo "Phase: $PHASE"
echo "======================================================================"

# --- Phase 0: Control Validation ---
if [[ "$PHASE" == "p0" || "$PHASE" == "all" ]]; then
  echo -e "\n>>> P0: Control validation"
  python experiments/01_control_validation/p0_control_experiment.py \
    --output-dir "$RESULTS/01_control"
fi

# --- Phase 1: Profession Expansion ---
if [[ "$PHASE" == "p1" || "$PHASE" == "all" ]]; then
  echo -e "\n>>> P1: Profession expansion"
  python experiments/02_function_word_analysis/p1_function_word_analysis.py \
    --output-dir "$RESULTS/02_function_word"
  python experiments/03_profession_expansion/p1_profession_expansion.py \
    --output-dir "$RESULTS/03_profession_expansion"
fi

# --- Phase 2: Name Confound Decomposition ---
if [[ "$PHASE" == "p2" || "$PHASE" == "all" ]]; then
  echo -e "\n>>> P2: Name confound decomposition"
  python experiments/04_name_confound/p2_name_confound.py \
    --output-dir "$RESULTS/04_name_confound"
fi

# --- Phase 2b: Rosenman Validation ---
if [[ "$PHASE" == "p2b" || "$PHASE" == "all" ]]; then
  echo -e "\n>>> P2b: Rosenman validation"
  python experiments/04_name_confound/p2b_rosenman_validation.py \
    --output-dir "$RESULTS/04_name_confound"
fi

# --- Real Text Validation ---
if [[ "$PHASE" == "rt" || "$PHASE" == "all" ]]; then
  echo -e "\n>>> RT: Real text validation"
  python experiments/05_real_text/real_text_validation.py \
    --output-dir "$RESULTS/05_real_text"
fi

# --- BM25 Control ---
if [[ "$PHASE" == "bm25" || "$PHASE" == "all" ]]; then
  echo -e "\n>>> BM25: Non-contextual control"
  python experiments/06_bm25_control/bm25_control.py \
    --output-dir "$RESULTS/06_bm25"
fi

# ==========================================================================
# NEW EXPERIMENTS (EMNLP Submission)
# ==========================================================================

# --- Exp 7a: Attention Ablation ---
if [[ "$PHASE" == "ablation" || "$PHASE" == "all" ]]; then
  echo -e "\n>>> Exp 7a: Attention ablation (masking + rollout + probing)"
  python experiments/07_attention_ablation/attention_ablation.py \
    --sub all --output-dir "$RESULTS/07_attention_ablation"
fi

# --- Exp 7b: Multi-Model Comparison ---
if [[ "$PHASE" == "multi" || "$PHASE" == "all" ]]; then
  echo -e "\n>>> Exp 7b: Multi-model comparison"
  python experiments/07_multi_model/multi_model_comparison.py \
    --output-dir "$RESULTS/07_multi_model"
fi

# --- Exp 8: MS MARCO Counterfactual ---
if [[ "$PHASE" == "msmarco" || "$PHASE" == "all" ]]; then
  echo -e "\n>>> Exp 8: MS MARCO counterfactual audit"
  python experiments/08_msmarco_counterfactual/msmarco_tcd_audit.py \
    --output-dir "$RESULTS/08_msmarco"
fi

# --- Exp 9: Ranking Impact ---
if [[ "$PHASE" == "ranking" || "$PHASE" == "all" ]]; then
  echo -e "\n>>> Exp 9: Ranking impact study"
  python experiments/09_ranking_impact/ranking_impact.py \
    --output-dir "$RESULTS/09_ranking_impact"
fi

# --- Exp 10: Mitigation ---
if [[ "$PHASE" == "mitigation" || "$PHASE" == "all" ]]; then
  echo -e "\n>>> Exp 10: Mitigation strategies"
  python experiments/10_mitigation/mitigation.py \
    --strategy all --output-dir "$RESULTS/10_mitigation"
fi

echo -e "\n======================================================================"
echo "✓ Pipeline complete: $PHASE"
echo "======================================================================"
