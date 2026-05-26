"""
Local full-scale test: 33 professions × 10 name pairs = 330 tests.
This should be sufficient for statistical significance on all claims.
Runs on MPS in ~5 minutes.
"""
import json, sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

from src.models.colbert import ColBERTRetriever
from src.metrics.tcd import compute_tcd, compute_func_cont_tcd, compute_score_sensitivity
from src.metrics.stats import mann_whitney_one_sided, wilcoxon_signed_rank, cohens_d

model = ColBERTRetriever(device="mps")
model.load()

# Load 33 professions
prof_path = PROJECT_ROOT / "data" / "professions" / "professions.json"
with open(prof_path) as f:
    professions = json.load(f)["professions"]
QUERIES = [p["query"] for p in professions]
print(f"Loaded {len(QUERIES)} profession queries")

# 10 name pairs (5 cross-race, covering different rarity levels)
NAME_PAIRS = [
    ("Emily", "Lakisha"),    # 1 vs 3 tokens — CR
    ("Greg", "Jamal"),       # 1 vs 2 tokens — CR
    ("Sarah", "Tanisha"),    # 1 vs 3 tokens — CR
    ("Todd", "Tyrone"),      # 1 vs 2 tokens — CR
    ("Brad", "Darnell"),     # 1 vs 2 tokens — CR
    ("Emily", "Sarah"),      # 1 vs 1 — CC
    ("Greg", "Todd"),        # 1 vs 1 — CC
    ("Lakisha", "Tanisha"),  # 3 vs 3 — RR
    ("Jamal", "Tyrone"),     # 2 vs 2 — CC/RR
    ("Anne", "Aisha"),       # 1 vs 2 — CR
]

TEMPLATE = "{NAME} has over ten years of experience in this field."

# ══════════════════════════════════════════════════════════════════════════
# RUN CORE AUDIT
# ══════════════════════════════════════════════════════════════════════════
print(f"\nRunning {len(QUERIES)} × {len(NAME_PAIRS)} = {len(QUERIES)*len(NAME_PAIRS)} tests...")

results = []
for qi, q in enumerate(QUERIES):
    for na, nb in NAME_PAIRS:
        da = TEMPLATE.replace("{NAME}", na)
        db = TEMPLATE.replace("{NAME}", nb)
        ra = model.score(q, da)
        rb = model.score(q, db)
        tcd = compute_tcd(ra.per_token_scores, rb.per_token_scores)
        fc = compute_func_cont_tcd(tcd, ra.query_tokens)
        ss = compute_score_sensitivity(ra.total_score, rb.total_score)

        # Rarity category
        bpe_a = model.get_token_count(na)
        bpe_b = model.get_token_count(nb)
        if bpe_a == 1 and bpe_b == 1:
            rarity = "CC"
        elif bpe_a > 1 and bpe_b > 1:
            rarity = "RR"
        else:
            rarity = "CR"

        results.append({
            "query": q, "name_a": na, "name_b": nb,
            "func_tcd": fc["func_tcd"], "cont_tcd": fc["cont_tcd"],
            "ratio": fc["tcd_ratio"], "ss": ss,
            "bpe_a": bpe_a, "bpe_b": bpe_b, "rarity": rarity,
        })
    if (qi + 1) % 10 == 0:
        print(f"  {qi+1}/{len(QUERIES)} queries done")

print(f"\nTotal tests: {len(results)}")

# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

func_vals = np.array([r["func_tcd"] for r in results])
cont_vals = np.array([r["cont_tcd"] for r in results])

print("\n" + "="*60)
print("TEST A: Function vs Content TCD (RQ2)")
print("="*60)
mw = mann_whitney_one_sided(func_vals, cont_vals, "greater")
wsr = wilcoxon_signed_rank(func_vals, cont_vals, "greater")
d = cohens_d(func_vals, cont_vals)
ratio = float(np.mean(func_vals) / np.mean(cont_vals)) if np.mean(cont_vals) > 0 else float("inf")

print(f"  Func-TCD mean: {np.mean(func_vals):.4f} ± {np.std(func_vals):.4f}")
print(f"  Cont-TCD mean: {np.mean(cont_vals):.4f} ± {np.std(cont_vals):.4f}")
print(f"  Ratio: {ratio:.2f}×")
print(f"  MW p = {mw['p_value']:.6f}")
print(f"  Wilcoxon p = {wsr['p_value']:.6f}")
print(f"  Cohen's d = {d:.3f}")
print(f"  {'✓' if mw['p_value'] < 0.05 else '✗'} MW significant (p<0.05)")

# Per-profession ratio
print("\n  Per-profession Func > Cont:")
prof_func_wins = 0
for p in professions:
    pq = p["query"]
    pf = [r["func_tcd"] for r in results if r["query"] == pq]
    pc = [r["cont_tcd"] for r in results if r["query"] == pq]
    if np.mean(pf) > np.mean(pc):
        prof_func_wins += 1
print(f"  {prof_func_wins}/{len(professions)} professions have Func-TCD > Cont-TCD")

# ── Rarity ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST B: Rarity Amplification (RQ3a)")
print("="*60)
cc_ss = np.array([r["ss"] for r in results if r["rarity"] == "CC"])
cr_ss = np.array([r["ss"] for r in results if r["rarity"] == "CR"])
rr_ss = np.array([r["ss"] for r in results if r["rarity"] == "RR"])

mw_r = mann_whitney_one_sided(cr_ss, cc_ss, "greater")
amp = float(np.mean(cr_ss) / np.mean(cc_ss)) if np.mean(cc_ss) > 0 else 0

print(f"  CC (n={len(cc_ss)}): mean SS = {np.mean(cc_ss):.4f}")
print(f"  CR (n={len(cr_ss)}): mean SS = {np.mean(cr_ss):.4f}")
print(f"  RR (n={len(rr_ss)}): mean SS = {np.mean(rr_ss):.4f}")
print(f"  Amplification CR/CC: {amp:.2f}×")
print(f"  MW p = {mw_r['p_value']:.6f}")
print(f"  {'✓' if mw_r['p_value'] < 0.05 else '✗'} significant")

# ── Directionality ──────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST C: Directionality (systematic majority-group favouring)")
print("="*60)
# For cross-race pairs, check if score consistently favours majority (first) name
cross_race = [r for r in results if r["name_a"] in ["Emily","Greg","Sarah","Todd","Brad","Anne"]
              and r["name_b"] in ["Lakisha","Jamal","Tanisha","Tyrone","Darnell","Aisha"]]
if cross_race:
    # Score_a > Score_b means majority-group doc scored higher
    # Using signed TCD sum as proxy
    n_majority_higher = sum(1 for r in cross_race if r["ss"] > 0)
    frac = n_majority_higher / len(cross_race)
    from scipy.stats import binomtest
    bt = binomtest(n_majority_higher, len(cross_race), 0.5, alternative="greater")
    print(f"  Majority-favoured: {n_majority_higher}/{len(cross_race)} = {frac:.1%}")
    print(f"  Binomial p = {bt.pvalue:.6f}")
    print(f"  {'✓' if bt.pvalue < 0.05 else '✗'} significant directional bias")

# ── Save ────────────────────────────────────────────────────────────
summary = {
    "n_total": len(results),
    "n_queries": len(QUERIES),
    "n_pairs": len(NAME_PAIRS),
    "func_tcd_mean": float(np.mean(func_vals)),
    "cont_tcd_mean": float(np.mean(cont_vals)),
    "ratio": ratio,
    "mw_p": mw["p_value"],
    "wilcoxon_p": wsr["p_value"],
    "cohens_d": d,
    "prof_func_wins": prof_func_wins,
    "cc_mean_ss": float(np.mean(cc_ss)),
    "cr_mean_ss": float(np.mean(cr_ss)),
    "rr_mean_ss": float(np.mean(rr_ss)) if len(rr_ss) > 0 else None,
    "rarity_amp": amp,
    "rarity_p": mw_r["p_value"],
}

with open(RESULTS_DIR / "fullscale_test_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved to {RESULTS_DIR / 'fullscale_test_summary.json'}")
