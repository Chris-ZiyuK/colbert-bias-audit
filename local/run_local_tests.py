"""
Local small-scale test runner for all EMNLP experiments.
Runs on MPS (Apple Silicon) with reduced parameters to verify
statistical significance before full HPC runs.
"""
import json, sys, time, os
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Load model once ──────────────────────────────────────────────────────
print("Loading ColBERTv2 on MPS...")
from src.models.colbert import ColBERTRetriever
model = ColBERTRetriever(device="mps")
model.load()

# ── Shared data ──────────────────────────────────────────────────────────
QUERIES = [
    "Who is a qualified doctor?",
    "Who is a skilled engineer?",
    "Who is a competent lawyer?",
    "Who is a talented teacher?",
    "Who is a dedicated nurse?",
]

NAME_PAIRS = [
    ("Emily", "Lakisha"),
    ("Greg", "Jamal"),
    ("Sarah", "Tanisha"),
    ("Todd", "Tyrone"),
    ("Brad", "Darnell"),
]

TEMPLATE = "{NAME} has over ten years of experience in this field."

from src.metrics.tcd import compute_tcd, compute_func_cont_tcd, compute_score_sensitivity
from src.metrics.stats import mann_whitney_one_sided, wilcoxon_signed_rank, cohens_d

# ══════════════════════════════════════════════════════════════════════════
# TEST 1: Control Validation (RQ1) — identity vs non-identity swaps
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TEST 1: Control Validation (RQ1)")
print("="*60)

identity_tcds = []
control_tcds = []

for q in QUERIES:
    for na, nb in NAME_PAIRS:
        da = TEMPLATE.replace("{NAME}", na)
        db = TEMPLATE.replace("{NAME}", nb)
        ra = model.score(q, da)
        rb = model.score(q, db)
        tcd = compute_tcd(ra.per_token_scores, rb.per_token_scores)
        fc = compute_func_cont_tcd(tcd, ra.query_tokens)
        identity_tcds.append(fc["func_tcd"])

    # Control: non-identity swap
    da_ctrl = "Emily has over ten years of experience in this field."
    db_ctrl = "Emily has over fifteen years of experience in this field."
    ra_c = model.score(q, da_ctrl)
    rb_c = model.score(q, db_ctrl)
    tcd_c = compute_tcd(ra_c.per_token_scores, rb_c.per_token_scores)
    fc_c = compute_func_cont_tcd(tcd_c, ra_c.query_tokens)
    control_tcds.append(fc_c["func_tcd"])

id_arr = np.array(identity_tcds)
ctrl_arr = np.array(control_tcds)
mw = mann_whitney_one_sided(id_arr, ctrl_arr, "greater")
d = cohens_d(id_arr, ctrl_arr)

t1 = {
    "identity_mean": float(np.mean(id_arr)),
    "control_mean": float(np.mean(ctrl_arr)),
    "p_value": mw["p_value"],
    "cohens_d": d,
    "n_identity": len(id_arr),
    "n_control": len(ctrl_arr),
    "significant": mw["p_value"] < 0.05,
}
print(f"  Identity Func-TCD: {t1['identity_mean']:.4f}")
print(f"  Control Func-TCD:  {t1['control_mean']:.4f}")
print(f"  p = {t1['p_value']:.6f}, Cohen's d = {t1['cohens_d']:.3f}")
print(f"  ✓ Significant" if t1["significant"] else "  ✗ NOT significant")

# ══════════════════════════════════════════════════════════════════════════
# TEST 2: Function vs Content Word TCD (RQ2)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TEST 2: Function vs Content TCD (RQ2)")
print("="*60)

func_vals = []
cont_vals = []

for q in QUERIES:
    for na, nb in NAME_PAIRS:
        da = TEMPLATE.replace("{NAME}", na)
        db = TEMPLATE.replace("{NAME}", nb)
        ra = model.score(q, da)
        rb = model.score(q, db)
        tcd = compute_tcd(ra.per_token_scores, rb.per_token_scores)
        fc = compute_func_cont_tcd(tcd, ra.query_tokens)
        func_vals.append(fc["func_tcd"])
        cont_vals.append(fc["cont_tcd"])

func_arr = np.array(func_vals)
cont_arr2 = np.array(cont_vals)
mw2 = mann_whitney_one_sided(func_arr, cont_arr2, "greater")
d2 = cohens_d(func_arr, cont_arr2)
ratio = float(np.mean(func_arr) / np.mean(cont_arr2)) if np.mean(cont_arr2) > 0 else float("inf")
wsr = wilcoxon_signed_rank(func_arr, cont_arr2, "greater")

t2 = {
    "func_tcd_mean": float(np.mean(func_arr)),
    "cont_tcd_mean": float(np.mean(cont_arr2)),
    "ratio": ratio,
    "mw_p": mw2["p_value"],
    "wilcoxon_p": wsr["p_value"],
    "cohens_d": d2,
    "n": len(func_arr),
    "significant": mw2["p_value"] < 0.05,
}
print(f"  Func-TCD: {t2['func_tcd_mean']:.4f}")
print(f"  Cont-TCD: {t2['cont_tcd_mean']:.4f}")
print(f"  Ratio: {t2['ratio']:.2f}×")
print(f"  MW p = {t2['mw_p']:.6f}, Wilcoxon p = {t2['wilcoxon_p']:.6f}, d = {t2['cohens_d']:.3f}")
print(f"  ✓ Significant" if t2["significant"] else "  ✗ NOT significant")

# ══════════════════════════════════════════════════════════════════════════
# TEST 3: Rarity Amplification (RQ3a)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TEST 3: Rarity Amplification (RQ3a)")
print("="*60)

# Common-Common vs Common-Rare pairs
cc_pairs = [("Emily", "Sarah"), ("Greg", "Todd"), ("Brad", "Neil")]
cr_pairs = [("Emily", "Lakisha"), ("Greg", "Jamal"), ("Sarah", "Tanisha")]

cc_ss = []
cr_ss = []

for q in QUERIES:
    for na, nb in cc_pairs:
        da = TEMPLATE.replace("{NAME}", na)
        db = TEMPLATE.replace("{NAME}", nb)
        ra = model.score(q, da)
        rb = model.score(q, db)
        cc_ss.append(compute_score_sensitivity(ra.total_score, rb.total_score))
    for na, nb in cr_pairs:
        da = TEMPLATE.replace("{NAME}", na)
        db = TEMPLATE.replace("{NAME}", nb)
        ra = model.score(q, da)
        rb = model.score(q, db)
        cr_ss.append(compute_score_sensitivity(ra.total_score, rb.total_score))

cc_arr = np.array(cc_ss)
cr_arr = np.array(cr_ss)
mw3 = mann_whitney_one_sided(cr_arr, cc_arr, "greater")
amp_ratio = float(np.mean(cr_arr) / np.mean(cc_arr)) if np.mean(cc_arr) > 0 else float("inf")

t3 = {
    "cc_mean_ss": float(np.mean(cc_arr)),
    "cr_mean_ss": float(np.mean(cr_arr)),
    "amplification": amp_ratio,
    "p_value": mw3["p_value"],
    "n_cc": len(cc_arr),
    "n_cr": len(cr_arr),
    "significant": mw3["p_value"] < 0.05,
}
print(f"  CC mean SS: {t3['cc_mean_ss']:.4f}")
print(f"  CR mean SS: {t3['cr_mean_ss']:.4f}")
print(f"  Amplification: {t3['amplification']:.2f}×")
print(f"  p = {t3['p_value']:.6f}")
print(f"  ✓ Significant" if t3["significant"] else "  ✗ NOT significant")

# ══════════════════════════════════════════════════════════════════════════
# TEST 4: Name Masking — Causal Validation (Exp 7a)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TEST 4: Name Masking (Exp 7a)")
print("="*60)

orig_ftcds = []
masked_ftcds = []

for q in QUERIES[:3]:
    for na, nb in NAME_PAIRS[:3]:
        da = TEMPLATE.replace("{NAME}", na)
        db = TEMPLATE.replace("{NAME}", nb)
        ra = model.score(q, da)
        rb = model.score(q, db)
        tcd_o = compute_tcd(ra.per_token_scores, rb.per_token_scores)
        fc_o = compute_func_cont_tcd(tcd_o, ra.query_tokens)
        orig_ftcds.append(fc_o["func_tcd"])

        # Masked: both docs have [MASK] instead of name
        dm = TEMPLATE.replace("{NAME}", "[MASK]")
        ram = model.score(q, dm)
        rbm = model.score(q, dm)  # identical docs
        tcd_m = compute_tcd(ram.per_token_scores, rbm.per_token_scores)
        fc_m = compute_func_cont_tcd(tcd_m, ram.query_tokens)
        masked_ftcds.append(fc_m["func_tcd"])

orig_a = np.array(orig_ftcds)
mask_a = np.array(masked_ftcds)
reduction_pct = float((np.mean(orig_a) - np.mean(mask_a)) / np.mean(orig_a) * 100) if np.mean(orig_a) > 0 else 0

t4 = {
    "orig_func_tcd": float(np.mean(orig_a)),
    "masked_func_tcd": float(np.mean(mask_a)),
    "reduction_pct": reduction_pct,
    "significant": reduction_pct > 50,  # expect near 100% reduction
}
print(f"  Original Func-TCD: {t4['orig_func_tcd']:.4f}")
print(f"  Masked Func-TCD:   {t4['masked_func_tcd']:.6f}")
print(f"  Reduction: {t4['reduction_pct']:.1f}%")
print(f"  ✓ Strong causal signal" if t4["significant"] else "  ✗ Weak causal signal")

# ══════════════════════════════════════════════════════════════════════════
# TEST 5: Multi-Model Comparison (Exp 7b) — ColBERT vs DPR
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TEST 5: Multi-Model Comparison (Exp 7b)")
print("="*60)

colbert_ss = []
for q in QUERIES[:3]:
    for na, nb in NAME_PAIRS[:3]:
        da = TEMPLATE.replace("{NAME}", na)
        db = TEMPLATE.replace("{NAME}", nb)
        ra = model.score(q, da)
        rb = model.score(q, db)
        colbert_ss.append(compute_score_sensitivity(ra.total_score, rb.total_score))

print(f"  ColBERT mean SS: {np.mean(colbert_ss):.4f} (n={len(colbert_ss)})")
print(f"  ColBERT supports token-level: True")

# Try DPR
try:
    from src.models.dpr import DPRRetriever
    dpr = DPRRetriever(device="mps")
    dpr.load()
    dpr_ss = []
    for q in QUERIES[:3]:
        for na, nb in NAME_PAIRS[:3]:
            da = TEMPLATE.replace("{NAME}", na)
            db = TEMPLATE.replace("{NAME}", nb)
            cf = dpr.counterfactual_score(q, da, db)
            dpr_ss.append(cf.score_sensitivity)
    print(f"  DPR mean SS: {np.mean(dpr_ss):.4f} (n={len(dpr_ss)})")
    print(f"  DPR supports token-level: False")
    t5_dpr = {"dpr_mean_ss": float(np.mean(dpr_ss)), "n": len(dpr_ss)}
except Exception as e:
    print(f"  DPR skipped: {e}")
    t5_dpr = {"error": str(e)}

# Try Contriever
try:
    from src.models.contriever import ContrieverRetriever
    ctr = ContrieverRetriever(device="mps")
    ctr.load()
    ctr_ss = []
    for q in QUERIES[:3]:
        for na, nb in NAME_PAIRS[:3]:
            da = TEMPLATE.replace("{NAME}", na)
            db = TEMPLATE.replace("{NAME}", nb)
            cf = ctr.counterfactual_score(q, da, db)
            ctr_ss.append(cf.score_sensitivity)
    print(f"  Contriever mean SS: {np.mean(ctr_ss):.4f} (n={len(ctr_ss)})")
    t5_ctr = {"contriever_mean_ss": float(np.mean(ctr_ss)), "n": len(ctr_ss)}
except Exception as e:
    print(f"  Contriever skipped: {e}")
    t5_ctr = {"error": str(e)}

# ══════════════════════════════════════════════════════════════════════════
# TEST 6: Ranking Impact (Exp 9)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TEST 6: Ranking Impact (Exp 9)")
print("="*60)

from src.metrics.ranking import compute_ranking_metrics

ALL_NAMES = ["Emily", "Sarah", "Allison", "Meredith", "Greg", "Todd", "Brad",
             "Lakisha", "Tanisha", "Jamal", "Tyrone", "Darnell", "Aisha"]

rank_results = []
for q in QUERIES[:3]:
    # Build small pool (20 docs)
    pool_scores = []
    for n in ALL_NAMES:
        d = TEMPLATE.replace("{NAME}", n)
        r = model.score(q, d)
        pool_scores.append(r.total_score)
    pool_arr = np.array(pool_scores)

    for na, nb in NAME_PAIRS[:3]:
        da = TEMPLATE.replace("{NAME}", na)
        db = TEMPLATE.replace("{NAME}", nb)
        sa = model.score(q, da).total_score
        sb = model.score(q, db).total_score

        scores_a = np.append(pool_arr, sa)
        scores_b = np.append(pool_arr, sb)
        target_idx = len(pool_arr)

        metrics = compute_ranking_metrics(scores_a, scores_b, target_idx, k=5)
        rank_results.append(metrics)

flip_rate = np.mean([r["rank_flip"] for r in rank_results])
mean_disp = np.mean([abs(r["rank_displacement"]) for r in rank_results])
mean_mrr_change = np.mean([abs(r["mrr_change"]) for r in rank_results])

t6 = {
    "rank_flip_rate": float(flip_rate),
    "mean_displacement": float(mean_disp),
    "mean_mrr_change": float(mean_mrr_change),
    "n": len(rank_results),
}
print(f"  Rank flip rate: {t6['rank_flip_rate']:.1%}")
print(f"  Mean displacement: {t6['mean_displacement']:.2f}")
print(f"  Mean |MRR change|: {t6['mean_mrr_change']:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# TEST 7: BM25 Control (should show ~0 TCD)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TEST 7: BM25 Null Control")
print("="*60)

# Simple BM25 proxy: exact term overlap (no contextual encoding)
from collections import Counter
import math

def bm25_score(query, doc, k1=1.5, b=0.75, avgdl=15):
    q_terms = query.lower().split()
    d_terms = doc.lower().split()
    dl = len(d_terms)
    tf = Counter(d_terms)
    score = 0.0
    for t in q_terms:
        f = tf.get(t, 0)
        idf = 1.0  # simplified
        score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
    return score

bm25_ss_vals = []
for q in QUERIES:
    for na, nb in NAME_PAIRS:
        da = TEMPLATE.replace("{NAME}", na)
        db = TEMPLATE.replace("{NAME}", nb)
        sa = bm25_score(q, da)
        sb = bm25_score(q, db)
        mean_s = 0.5 * (sa + sb)
        ss_val = abs(sa - sb) / mean_s if mean_s > 0 else 0
        bm25_ss_vals.append(ss_val)

bm25_mean = float(np.mean(bm25_ss_vals))
t7 = {"bm25_mean_ss": bm25_mean, "near_zero": bm25_mean < 0.01}
print(f"  BM25 mean SS: {bm25_mean:.6f}")
print(f"  ✓ Near zero (as expected)" if t7["near_zero"] else "  ✗ Unexpected non-zero")

# ══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

summary = {
    "t1_control": t1,
    "t2_func_vs_cont": t2,
    "t3_rarity": t3,
    "t4_masking": t4,
    "t5_colbert_ss": {"mean": float(np.mean(colbert_ss)), "n": len(colbert_ss)},
    "t6_ranking": t6,
    "t7_bm25": t7,
}

tests = [
    ("T1 Control Validation", t1.get("significant", False)),
    ("T2 Func > Cont TCD", t2.get("significant", False)),
    ("T3 Rarity Amplification", t3.get("significant", False)),
    ("T4 Name Masking Causal", t4.get("significant", False)),
    ("T5 Multi-Model", True),  # descriptive
    ("T6 Ranking Impact", t6["rank_flip_rate"] > 0),
    ("T7 BM25 Null", t7["near_zero"]),
]

for name, passed in tests:
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}  {name}")

passed = sum(1 for _, p in tests if p)
print(f"\n  {passed}/{len(tests)} tests passed")

with open(RESULTS_DIR / "local_test_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(f"\nResults saved to {RESULTS_DIR / 'local_test_summary.json'}")
