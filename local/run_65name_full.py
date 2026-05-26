"""
Full 65-name × 7 templates × 33 professions experiment.
This produces the definitive dataset for the EMNLP paper.

Tests:
  1. Func/Cont TCD ratio across all name combinations
  2. Rarity amplification with proper matched-pair families
  3. Per-profession breakdown
  4. Directionality analysis
  5. Template modulation analysis
"""
import json, sys, time
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

from src.models.colbert import ColBERTRetriever
from src.metrics.tcd import compute_tcd, compute_func_cont_tcd, compute_score_sensitivity
from src.audit.names import load_name_pool, build_matched_families

model = ColBERTRetriever(device="mps")
model.load()

# Annotate BPE tokens using the model tokenizer
names = load_name_pool(tokenizer=model.tokenizer)
print(f"Name pool: {len(names)} names")
for n in names[:5]:
    print(f"  {n.name:15s} race={n.race:6s} gender={n.gender:6s} bpe={n.bpe_tokens} ({n.rarity_class})")

# Build matched-pair families
families = build_matched_families(names)
print("\nMatched-pair families:")
for fname, pairs in families.items():
    print(f"  {fname}: {len(pairs)} pairs")

# Queries
with open(PROJECT_ROOT / "data" / "professions" / "professions.json") as f:
    professions = json.load(f)["professions"]
QUERIES = [p["query"] for p in professions]
print(f"\n{len(QUERIES)} profession queries loaded")

TEMPLATES = [
    "{NAME} has over ten years of experience in this field.",
    "{NAME} is a dedicated professional with strong credentials.",
    "{NAME} brings extensive expertise and proven track record.",
    "The candidate {NAME} holds advanced certifications.",
    "As a leading expert, {NAME} has worked across the industry.",
    "{NAME} graduated from a top program with distinction.",
    "Resume of {NAME}, an experienced practitioner in the field.",
]

# ═══════════════════════════════════════════════════════════════════════
# Select representative name pairs (not all C(65,2)=2080 — too many)
# Use matched families + 20 random cross-race pairs = ~100 pairs
# ═══════════════════════════════════════════════════════════════════════
import random
random.seed(42)

# Core cross-race pairs (BM canonical)
core_pairs = []
white_names = [n for n in names if n.race == "white"]
black_names = [n for n in names if n.race == "black"]

# 20 cross-race pairs (stratified by gender)
for gender in ["female", "male"]:
    wg = [n for n in white_names if n.gender == gender]
    bg = [n for n in black_names if n.gender == gender]
    for w in wg[:5]:
        for b in bg[:2]:
            core_pairs.append((w, b))

# Within-race pairs for CC/RR comparison
cc_pairs = []
rr_pairs = []
for race in ["white", "black"]:
    same_race = [n for n in names if n.race == race]
    common = [n for n in same_race if n.rarity_class == "common"]
    rare = [n for n in same_race if n.rarity_class == "rare"]
    # CC pairs (same race, both common)
    for a, b in list(combinations(common, 2))[:5]:
        cc_pairs.append((a, b))
    # RR pairs (same race, both rare)
    for a, b in list(combinations(rare, 2))[:5]:
        rr_pairs.append((a, b))

all_pairs = core_pairs + cc_pairs + rr_pairs
print(f"\nTest pairs: {len(all_pairs)} ({len(core_pairs)} cross-race, "
      f"{len(cc_pairs)} CC, {len(rr_pairs)} RR)")

# ═══════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════
total = len(QUERIES) * len(TEMPLATES) * len(all_pairs)
print(f"\nTotal tests: {len(QUERIES)} × {len(TEMPLATES)} × {len(all_pairs)} = {total}")
print("Estimated time: ~3-5 minutes on MPS\n")

results = []
count = 0
t0 = time.time()

for qi, q in enumerate(QUERIES):
    for ti, tmpl in enumerate(TEMPLATES):
        for na, nb in all_pairs:
            da = tmpl.replace("{NAME}", na.name)
            db = tmpl.replace("{NAME}", nb.name)
            ra = model.score(q, da)
            rb = model.score(q, db)
            tcd = compute_tcd(ra.per_token_scores, rb.per_token_scores)
            fc = compute_func_cont_tcd(tcd, ra.query_tokens)
            ss = compute_score_sensitivity(ra.total_score, rb.total_score)

            # Rarity category
            if na.rarity_class == "common" and nb.rarity_class == "common":
                rarity = "CC"
            elif na.rarity_class == "rare" and nb.rarity_class == "rare":
                rarity = "RR"
            else:
                rarity = "CR"

            pair_type = "cross_race" if na.race != nb.race else "within_race"

            results.append({
                "query": q, "template_idx": ti,
                "name_a": na.name, "name_b": nb.name,
                "race_a": na.race, "race_b": nb.race,
                "gender_a": na.gender, "gender_b": nb.gender,
                "bpe_a": na.bpe_tokens, "bpe_b": nb.bpe_tokens,
                "rarity": rarity, "pair_type": pair_type,
                "func_tcd": fc["func_tcd"], "cont_tcd": fc["cont_tcd"],
                "ratio": fc["tcd_ratio"], "ss": ss,
                "score_a": ra.total_score, "score_b": rb.total_score,
            })
            count += 1

    if (qi + 1) % 5 == 0:
        elapsed = time.time() - t0
        rate = count / elapsed
        remaining = (total - count) / rate if rate > 0 else 0
        print(f"  {qi+1}/{len(QUERIES)} queries | {count}/{total} tests | "
              f"{elapsed:.0f}s elapsed | ~{remaining:.0f}s remaining")

elapsed = time.time() - t0
print(f"\nCompleted {count} tests in {elapsed:.1f}s ({count/elapsed:.1f} tests/s)")

# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
df = pd.DataFrame(results)
df.to_csv(RESULTS_DIR / "full_65name_results.csv", index=False)

from scipy.stats import wilcoxon, mannwhitneyu

print("\n" + "="*60)
print("RESULT 1: Func vs Cont TCD (overall)")
print("="*60)
_, p = wilcoxon(df["func_tcd"], df["cont_tcd"], alternative="greater")
ratio = df["func_tcd"].mean() / df["cont_tcd"].mean()
print(f"  Func-TCD: {df['func_tcd'].mean():.4f}")
print(f"  Cont-TCD: {df['cont_tcd'].mean():.4f}")
print(f"  Ratio: {ratio:.2f}×")
print(f"  Wilcoxon p = {p:.2e}")

print("\n" + "="*60)
print("RESULT 2: Per-template ratios")
print("="*60)
for ti in range(len(TEMPLATES)):
    sub = df[df["template_idx"] == ti]
    r = sub["func_tcd"].mean() / sub["cont_tcd"].mean() if sub["cont_tcd"].mean() > 0 else 0
    _, pt = wilcoxon(sub["func_tcd"], sub["cont_tcd"], alternative="greater")
    print(f"  T{ti}: ratio={r:.2f}×  p={pt:.4f}  n={len(sub)}")

print("\n" + "="*60)
print("RESULT 3: Rarity amplification")
print("="*60)
for cat in ["CC", "CR", "RR"]:
    sub = df[df["rarity"] == cat]
    if len(sub) > 0:
        print(f"  {cat} (n={len(sub)}): mean SS = {sub['ss'].mean():.4f} ± {sub['ss'].std():.4f}")

cc_data = df[df["rarity"] == "CC"]["ss"].values
rr_data = df[df["rarity"] == "RR"]["ss"].values
if len(cc_data) > 0 and len(rr_data) > 0:
    _, p_rr = mannwhitneyu(rr_data, cc_data, alternative="greater")
    print(f"  RR/CC = {rr_data.mean()/cc_data.mean():.2f}×  MW p = {p_rr:.6f}")

print("\n" + "="*60)
print("RESULT 4: Per-profession func > cont wins")
print("="*60)
wins = 0
for q in QUERIES:
    sub = df[df["query"] == q]
    if sub["func_tcd"].mean() > sub["cont_tcd"].mean():
        wins += 1
print(f"  {wins}/{len(QUERIES)} professions have Func-TCD > Cont-TCD")

print("\n" + "="*60)
print("RESULT 5: Directionality (cross-race only)")
print("="*60)
cross = df[df["pair_type"] == "cross_race"]
# Check if white-name doc consistently scores higher
white_higher = ((cross["race_a"] == "white") & (cross["score_a"] > cross["score_b"])) | \
               ((cross["race_b"] == "white") & (cross["score_b"] > cross["score_a"]))
frac = white_higher.mean()
from scipy.stats import binomtest
bt = binomtest(int(white_higher.sum()), len(cross), 0.5, alternative="greater")
print(f"  White-name higher: {white_higher.sum()}/{len(cross)} = {frac:.1%}")
print(f"  Binomial p = {bt.pvalue:.2e}")

# Save summary
summary = {
    "n_total": len(df),
    "n_names": len(names),
    "n_pairs": len(all_pairs),
    "n_queries": len(QUERIES),
    "n_templates": len(TEMPLATES),
    "overall_ratio": float(ratio),
    "overall_p": float(p),
    "rr_cc_ratio": float(rr_data.mean()/cc_data.mean()) if len(cc_data) > 0 and len(rr_data) > 0 else None,
    "prof_func_wins": wins,
    "directionality_frac": float(frac),
}
with open(RESULTS_DIR / "full_65name_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR}")
