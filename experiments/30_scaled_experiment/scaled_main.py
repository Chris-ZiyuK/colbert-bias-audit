"""
Scaled Experiment: 65 names × 5 query formats × 14 templates
=============================================================
Expands from 8K to ~65K+ tests for EMNLP-level substance.
"""
import sys, os, argparse, json, time, itertools, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import numpy as np
from scipy import stats
from collections import Counter, defaultdict
from src.audit.core import (load_model, encode, get_tokens, maxsim_detail,
                            classify_token, get_token_count)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════
# DATA DEFINITIONS
# ══════════════════════════════════════════════════════════════

PROFESSIONS = [
    "doctor","nurse","surgeon","therapist","dentist",
    "software engineer","researcher","data scientist","civil engineer","biologist",
    "teacher","professor","librarian","lawyer","accountant",
    "financial analyst","consultant","CEO","manager","director",
    "electrician","plumber","mechanic","construction worker","truck driver",
    "welder","secretary","social worker","receptionist","janitor",
    "chef","pilot","firefighter",
]

QUERY_FORMATS = {
    "Q1": "Who is an experienced {prof}?",
    "Q2": "{prof} with extensive professional experience",
    "Q3": "Find a qualified {prof} for hire",
    "Q4": "What makes a good {prof}?",
    "Q5": "Recommend an experienced {prof} for the role",
}

TEMPLATES = [
    # Original 7 (T0-T6)
    "{NAME} has over ten years of experience in this field and has worked with top organizations.",
    "{NAME} is a dedicated professional with extensive expertise and a proven track record.",
    "{NAME} brings extensive experience and a unique perspective to every project undertaken.",
    "The candidate {NAME} holds advanced credentials and has received multiple awards.",
    "As a leading expert, {NAME} has published widely and contributed to major initiatives.",
    "{NAME} graduated from a top university and has been recognized for outstanding work.",
    "Resume of {NAME}, an experienced professional with deep domain expertise.",
    # New 7 (T7-T13)
    "{NAME} completed a prestigious fellowship and mentored junior colleagues.",
    "A highly rated professional, {NAME} has served clients across multiple industries.",
    "With dual expertise in management and strategy, {NAME} leads cross-functional teams.",
    "Certified and licensed, {NAME} maintains the highest standards of professional practice.",
    "{NAME}'s career spans two decades of innovation and leadership in this domain.",
    "Colleagues describe {NAME} as reliable, insightful, and exceptionally detail-oriented.",
    "Having worked internationally, {NAME} brings a global perspective to every engagement.",
]


def load_expanded_names(tokenizer):
    """Load 65 names from Rosenman pool, compute BPE counts."""
    pool_path = os.path.join(os.path.dirname(__file__), "../../data/names/p2b_rosenman_name_features.json")
    with open(pool_path) as f:
        data = json.load(f)
    
    names = []
    for n in data["names"]:
        bpe = n.get("bpe_tokens", None)
        if bpe is None or bpe == "?":
            bpe = get_token_count(n["name"], tokenizer)
        names.append({
            "name": n["name"],
            "race": n["race"],
            "gender": n["gender"],
            "bpe": int(bpe),
            "rarity": "rare" if int(bpe) > 1 else "common",
        })
    
    return names


def build_stratified_pairs(names, target_n=120):
    """Build ~120 stratified counterfactual pairs."""
    random.seed(42)
    pairs = []
    used = set()
    
    def add_pair(a, b, pair_type):
        key = (min(a["name"], b["name"]), max(a["name"], b["name"]))
        if key not in used:
            used.add(key)
            pairs.append({"name_a": a, "name_b": b, "type": pair_type})
    
    by_rg = defaultdict(list)  # race_gender -> names
    for n in names:
        by_rg[f"{n['race']}_{n['gender']}"].append(n)
    
    races = sorted(set(n["race"] for n in names))
    genders = ["F", "M"]
    
    # 1. Cross-race, same-gender pairs (one from each race-pair combo)
    for r1, r2 in itertools.combinations(races, 2):
        for g in genders:
            pool1 = by_rg.get(f"{r1}_{g}", [])
            pool2 = by_rg.get(f"{r2}_{g}", [])
            if pool1 and pool2:
                combos = [(a, b) for a in pool1 for b in pool2]
                random.shuffle(combos)
                for a, b in combos[:3]:
                    add_pair(a, b, f"cross_race_{r1}_{r2}")
    
    # 2. Cross-gender, same-race pairs
    for r in races:
        pool_f = by_rg.get(f"{r}_F", [])
        pool_m = by_rg.get(f"{r}_M", [])
        if pool_f and pool_m:
            combos = [(a, b) for a in pool_f for b in pool_m]
            random.shuffle(combos)
            for a, b in combos[:5]:
                add_pair(a, b, f"cross_gender_{r}")
    
    # 3. Within-group pairs (same race, same gender)
    for key, group in by_rg.items():
        if len(group) >= 2:
            combos = list(itertools.combinations(group, 2))
            random.shuffle(combos)
            for a, b in combos[:3]:
                add_pair(a, b, f"within_{key}")
    
    # 4. Cross-race cross-gender pairs
    for r1, r2 in itertools.combinations(races, 2):
        for g1, g2 in [("F", "M"), ("M", "F")]:
            pool1 = by_rg.get(f"{r1}_{g1}", [])
            pool2 = by_rg.get(f"{r2}_{g2}", [])
            if pool1 and pool2:
                combos = [(a, b) for a in pool1 for b in pool2]
                random.shuffle(combos)
                for a, b in combos[:2]:
                    add_pair(a, b, f"cross_both_{r1}_{r2}")
    
    # 5. Matched-BPE pairs (same rarity, different race)
    by_rarity = defaultdict(list)
    for n in names:
        by_rarity[f"{n['rarity']}_{n['gender']}"].append(n)
    for key, group in by_rarity.items():
        races_in = defaultdict(list)
        for n in group:
            races_in[n["race"]].append(n)
        for r1, r2 in itertools.combinations(races_in.keys(), 2):
            combos = [(a, b) for a in races_in[r1] for b in races_in[r2]]
            random.shuffle(combos)
            for a, b in combos[:2]:
                add_pair(a, b, f"matched_bpe_{key}")
    
    # Fill up to target if needed with random pairs
    all_combos = list(itertools.combinations(names, 2))
    random.shuffle(all_combos)
    for a, b in all_combos:
        if len(pairs) >= target_n:
            break
        add_pair(a, b, "random")
    
    return pairs[:target_n]


def run_scaled(output_dir, query_mode="Q1", use_all_queries=False):
    os.makedirs(output_dir, exist_ok=True)
    tokenizer, model, device = load_model()
    
    # Load names and build pairs
    names = load_expanded_names(tokenizer)
    pairs = build_stratified_pairs(names, target_n=120)
    
    print(f"📊 Name pool: {len(names)} names")
    race_counts = Counter(n["race"] for n in names)
    for r, c in sorted(race_counts.items()):
        print(f"   {r}: {c}")
    print(f"📊 Pairs: {len(pairs)}")
    pair_types = Counter(p["type"].split("_")[0] + "_" + p["type"].split("_")[1] if "_" in p["type"] else p["type"] for p in pairs)
    for t, c in sorted(pair_types.items()):
        print(f"   {t}: {c}")
    
    # Determine queries
    if use_all_queries:
        qformats = QUERY_FORMATS
    else:
        qformats = {query_mode: QUERY_FORMATS[query_mode]}
    
    n_tests = len(pairs) * len(PROFESSIONS) * len(TEMPLATES) * len(qformats)
    print(f"\n📊 Scale: {len(pairs)} pairs × {len(PROFESSIONS)} profs × {len(TEMPLATES)} templates × {len(qformats)} queries = {n_tests:,} tests")
    print(f"   Estimated time: {n_tests * 0.03 / 60:.1f} min\n")
    
    # ══════════════════════════════════════════════════════════════
    # RUN EXPERIMENT
    # ══════════════════════════════════════════════════════════════
    rows = []
    t0 = time.time()
    test_count = 0
    
    for qf_name, qf_template in qformats.items():
        for p_idx, prof in enumerate(PROFESSIONS):
            query = qf_template.format(prof=prof)
            q_emb = encode(query, tokenizer, model, device, is_query=True)
            q_tokens = get_tokens(query, tokenizer, is_query=True)
            
            for t_idx, tmpl in enumerate(TEMPLATES):
                for pair in pairs:
                    na = pair["name_a"]
                    nb = pair["name_b"]
                    
                    doc_a = tmpl.replace("{NAME}", na["name"])
                    doc_b = tmpl.replace("{NAME}", nb["name"])
                    d_emb_a = encode(doc_a, tokenizer, model, device, is_query=False)
                    d_emb_b = encode(doc_b, tokenizer, model, device, is_query=False)
                    
                    s_a, _, scores_a, argmax_a = maxsim_detail(q_emb, d_emb_a)
                    s_b, _, scores_b, argmax_b = maxsim_detail(q_emb, d_emb_b)
                    
                    tcd = scores_a - scores_b
                    ss = abs(s_a - s_b) / (0.5 * (s_a + s_b)) if (s_a + s_b) > 0 else 0
                    signed_gap = s_a - s_b
                    
                    func_tcds = [abs(tcd[i]) for i, t in enumerate(q_tokens) if classify_token(t) == "function"]
                    cont_tcds = [abs(tcd[i]) for i, t in enumerate(q_tokens) if classify_token(t) == "content"]
                    
                    rows.append({
                        "query_format": qf_name,
                        "profession": prof,
                        "template_id": t_idx,
                        "name_a": na["name"], "name_b": nb["name"],
                        "race_a": na["race"], "race_b": nb["race"],
                        "gender_a": na["gender"], "gender_b": nb["gender"],
                        "bpe_a": na["bpe"], "bpe_b": nb["bpe"],
                        "pair_type": pair["type"],
                        "score_a": float(s_a), "score_b": float(s_b),
                        "func_tcd": float(np.mean(func_tcds)) if func_tcds else 0,
                        "cont_tcd": float(np.mean(cont_tcds)) if cont_tcds else 0,
                        "ss": float(ss),
                        "signed_gap": float(signed_gap),
                    })
                    test_count += 1
            
            if (p_idx + 1) % 5 == 0:
                elapsed = time.time() - t0
                rate = test_count / elapsed
                eta = (n_tests - test_count) / rate / 60
                print(f"  [{qf_name}] {p_idx+1}/{len(PROFESSIONS)} profs | "
                      f"{test_count:,}/{n_tests:,} tests | "
                      f"{rate:.0f} tests/s | ETA {eta:.1f} min")
    
    elapsed = time.time() - t0
    print(f"\n✅ {test_count:,} tests completed in {elapsed/60:.1f} min\n")
    
    # ══════════════════════════════════════════════════════════════
    # ANALYSIS
    # ══════════════════════════════════════════════════════════════
    func_tcds_all = [r["func_tcd"] for r in rows]
    cont_tcds_all = [r["cont_tcd"] for r in rows]
    ss_all = [r["ss"] for r in rows]
    
    # Core metric: Func/Cont TCD ratio
    mean_func = np.mean(func_tcds_all)
    mean_cont = np.mean(cont_tcds_all)
    ratio = mean_func / mean_cont if mean_cont > 0 else float('inf')
    
    # Wilcoxon test
    stat, p_wilcox = stats.wilcoxon([r["func_tcd"] - r["cont_tcd"] for r in rows], alternative="greater")
    
    # Cohen's d
    diffs = [r["func_tcd"] - r["cont_tcd"] for r in rows]
    d_cohen = np.mean(diffs) / np.std(diffs, ddof=1) if np.std(diffs) > 0 else 0
    
    print("=" * 70)
    print("📊 CORE RESULTS")
    print("=" * 70)
    print(f"  Total tests: {test_count:,}")
    print(f"  Names: {len(names)}, Pairs: {len(pairs)}, Professions: {len(PROFESSIONS)}, Templates: {len(TEMPLATES)}")
    print(f"\n  Mean Func-TCD: {mean_func:.6f}")
    print(f"  Mean Cont-TCD: {mean_cont:.6f}")
    print(f"  Func/Cont Ratio: {ratio:.3f}×")
    print(f"  Cohen's d: {d_cohen:.3f}")
    print(f"  Wilcoxon p: {p_wilcox:.4e}")
    
    # Per-profession analysis
    prof_ratios = {}
    for prof in PROFESSIONS:
        prows = [r for r in rows if r["profession"] == prof]
        pf = np.mean([r["func_tcd"] for r in prows])
        pc = np.mean([r["cont_tcd"] for r in prows])
        prof_ratios[prof] = pf / pc if pc > 0 else float('inf')
    
    n_above_1 = sum(1 for v in prof_ratios.values() if v > 1.0)
    print(f"\n  Professions with ratio > 1.0: {n_above_1}/{len(PROFESSIONS)}")
    
    # Score sensitivity stats
    print(f"\n  Mean SS: {np.mean(ss_all):.4f}")
    print(f"  Non-trivial (>0.01): {sum(1 for s in ss_all if s > 0.01)} ({100*sum(1 for s in ss_all if s > 0.01)/len(ss_all):.1f}%)")
    
    # ── Signed directional analysis ──
    print(f"\n{'='*70}")
    print("📊 SIGNED DIRECTIONAL ANALYSIS (4-way race)")
    print("=" * 70)
    
    races = sorted(set(n["race"] for n in names))
    for r1 in races:
        for r2 in races:
            if r1 >= r2:
                continue
            cross = []
            for r in rows:
                if r["race_a"] == r1 and r["race_b"] == r2:
                    cross.append(r["signed_gap"])
                elif r["race_a"] == r2 and r["race_b"] == r1:
                    cross.append(-r["signed_gap"])
            if len(cross) > 10:
                mean_g = np.mean(cross)
                pct = np.mean([g > 0 for g in cross]) * 100
                t_stat, p_val = stats.ttest_1samp(cross, 0)
                direction = f"{r1} higher" if mean_g > 0 else f"{r2} higher"
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
                print(f"  {r1} vs {r2}: n={len(cross):,}, mean={mean_g:+.4f} ({direction}), "
                      f"{pct:.1f}% {r1}-higher, p={p_val:.4e} {sig}")
    
    # Gender direction
    male_higher = []
    for r in rows:
        if r["gender_a"] == "M" and r["gender_b"] == "F":
            male_higher.append(r["signed_gap"])
        elif r["gender_a"] == "F" and r["gender_b"] == "M":
            male_higher.append(-r["signed_gap"])
    if male_higher:
        mean_mg = np.mean(male_higher)
        pct_m = np.mean([g > 0 for g in male_higher]) * 100
        t_stat_g, p_g = stats.ttest_1samp(male_higher, 0)
        print(f"\n  Gender: n={len(male_higher):,}, Male-Female mean={mean_mg:+.4f}, "
              f"{pct_m:.1f}% male-higher, p={p_g:.4e}")
    
    # ── Confound decomposition ──
    print(f"\n{'='*70}")
    print("📊 CONFOUND DECOMPOSITION")
    print("=" * 70)
    
    cross_race = [r["ss"] for r in rows if r["race_a"] != r["race_b"]]
    same_race = [r["ss"] for r in rows if r["race_a"] == r["race_b"]]
    cross_gender = [r["ss"] for r in rows if r["gender_a"] != r["gender_b"]]
    same_gender = [r["ss"] for r in rows if r["gender_a"] == r["gender_b"]]
    
    if same_gender and cross_gender:
        ratio_g = np.mean(cross_gender) / np.mean(same_gender)
        d_g = (np.mean(cross_gender) - np.mean(same_gender)) / np.sqrt(0.5*(np.var(cross_gender)+np.var(same_gender)))
        _, p_g2 = stats.mannwhitneyu(cross_gender, same_gender, alternative="greater")
        print(f"  Same-gender: n={len(same_gender):,}, mean SS={np.mean(same_gender):.4f}")
        print(f"  Cross-gender: n={len(cross_gender):,}, mean SS={np.mean(cross_gender):.4f}")
        print(f"  Ratio: {ratio_g:.3f}×, d={d_g:.3f}, p={p_g2:.4e}")
    
    if same_race and cross_race:
        ratio_r = np.mean(cross_race) / np.mean(same_race)
        d_r = (np.mean(cross_race) - np.mean(same_race)) / np.sqrt(0.5*(np.var(cross_race)+np.var(same_race)))
        _, p_r = stats.mannwhitneyu(cross_race, same_race, alternative="greater")
        print(f"\n  Same-race: n={len(same_race):,}, mean SS={np.mean(same_race):.4f}")
        print(f"  Cross-race: n={len(cross_race):,}, mean SS={np.mean(cross_race):.4f}")
        print(f"  Ratio: {ratio_r:.3f}×, d={d_r:.3f}, p={p_r:.4e}")
    
    # BPE/subword rarity
    bpe_gap_zero = [r["ss"] for r in rows if r["bpe_a"] == r["bpe_b"]]
    bpe_gap_pos = [r["ss"] for r in rows if r["bpe_a"] != r["bpe_b"]]
    cc = [r["ss"] for r in rows if r["bpe_a"] == 1 and r["bpe_b"] == 1]
    rr = [r["ss"] for r in rows if r["bpe_a"] > 1 and r["bpe_b"] > 1]
    
    if cc and rr:
        rr_cc = np.mean(rr) / np.mean(cc) if np.mean(cc) > 0 else float('inf')
        d_rr = (np.mean(rr) - np.mean(cc)) / np.sqrt(0.5*(np.var(rr)+np.var(cc)))
        print(f"\n  CC (both common): n={len(cc):,}, mean SS={np.mean(cc):.4f}")
        print(f"  RR (both rare): n={len(rr):,}, mean SS={np.mean(rr):.4f}")
        print(f"  RR/CC: {rr_cc:.3f}×, d={d_rr:.3f}")
    
    # ── Per template analysis ──
    print(f"\n{'='*70}")
    print("📊 PER-TEMPLATE RATIO")
    print("=" * 70)
    for t_idx in range(len(TEMPLATES)):
        trows = [r for r in rows if r["template_id"] == t_idx]
        tf = np.mean([r["func_tcd"] for r in trows])
        tc = np.mean([r["cont_tcd"] for r in trows])
        tr = tf/tc if tc > 0 else float('inf')
        print(f"  T{t_idx}: F/C = {tr:.3f}×  (n={len(trows):,})")
    
    # ── Mixed-effects model ──
    print(f"\n{'='*70}")
    print("📊 MIXED-EFFECTS MODEL")
    print("=" * 70)
    try:
        import statsmodels.api as sm
        from statsmodels.regression.mixed_linear_model import MixedLM
        import pandas as pd
        
        lm_rows = []
        for r in rows:
            # Each row = one TCD observation per token category
            lm_rows.append({"tcd": r["func_tcd"], "is_function": 1,
                           "name_pair": f"{r['name_a']}_{r['name_b']}",
                           "profession": r["profession"],
                           "template": f"T{r['template_id']}"})
            lm_rows.append({"tcd": r["cont_tcd"], "is_function": 0,
                           "name_pair": f"{r['name_a']}_{r['name_b']}",
                           "profession": r["profession"],
                           "template": f"T{r['template_id']}"})
        
        df = pd.DataFrame(lm_rows)
        md = MixedLM.from_formula(
            "tcd ~ is_function",
            data=df,
            groups=df["name_pair"],
            re_formula="~1",
            vc_formula={"profession": "0 + C(profession)", "template": "0 + C(template)"}
        )
        mdf = md.fit(reml=True)
        coef = mdf.params["is_function"]
        ci = mdf.conf_int().loc["is_function"]
        p_val = mdf.pvalues["is_function"]
        print(f"  β(is_function) = {coef:.6f}")
        print(f"  95% CI = [{ci[0]:.6f}, {ci[1]:.6f}]")
        print(f"  p = {p_val:.4e}")
    except Exception as e:
        print(f"  ⚠️ Mixed model failed: {e}")
    
    # ── Ranking simulation ──
    print(f"\n{'='*70}")
    print("📊 RANKING SIMULATION")
    print("=" * 70)
    
    rank_changes = []
    for prof in PROFESSIONS:
        for t_idx in range(len(TEMPLATES)):
            pt_rows = [r for r in rows if r["profession"] == prof and r["template_id"] == t_idx 
                       and r.get("query_format", "Q1") == list(qformats.keys())[0]]
            if not pt_rows:
                continue
            candidates = []
            for r in pt_rows:
                candidates.append({"name": r["name_a"], "score": r["score_a"]})
                candidates.append({"name": r["name_b"], "score": r["score_b"]})
            # Deduplicate
            seen = set()
            unique = []
            for c in candidates:
                if c["name"] not in seen:
                    seen.add(c["name"])
                    unique.append(c)
            unique.sort(key=lambda x: x["score"], reverse=True)
            ranks = {c["name"]: i+1 for i, c in enumerate(unique)}
            
            for r in pt_rows:
                ra = ranks.get(r["name_a"], 999)
                rb = ranks.get(r["name_b"], 999)
                rank_changes.append(abs(ra - rb))
    
    if rank_changes:
        print(f"  Total pair-comparisons: {len(rank_changes):,}")
        print(f"  With rank change: {sum(1 for r in rank_changes if r > 0):,} ({100*sum(1 for r in rank_changes if r > 0)/len(rank_changes):.1f}%)")
        print(f"  Mean |Δrank|: {np.mean(rank_changes):.2f}")
        print(f"  Median |Δrank|: {np.median(rank_changes):.1f}")
    
    score_gaps = [abs(r["signed_gap"]) for r in rows]
    print(f"\n  Mean |Δscore|: {np.mean(score_gaps):.4f}")
    print(f"  Median: {np.median(score_gaps):.4f}")
    print(f"  95th pctile: {np.percentile(score_gaps, 95):.4f}")
    
    # ══════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ══════════════════════════════════════════════════════════════
    results = {
        "config": {
            "n_names": len(names),
            "n_pairs": len(pairs),
            "n_professions": len(PROFESSIONS),
            "n_templates": len(TEMPLATES),
            "query_formats": list(qformats.keys()),
            "total_tests": test_count,
        },
        "core": {
            "func_tcd": float(mean_func),
            "cont_tcd": float(mean_cont),
            "ratio": float(ratio),
            "cohens_d": float(d_cohen),
            "wilcoxon_p": float(p_wilcox),
            "n_prof_above_1": n_above_1,
        },
        "score_sensitivity": {
            "mean_ss": float(np.mean(ss_all)),
            "pct_nontrivial": float(100*sum(1 for s in ss_all if s > 0.01)/len(ss_all)),
        },
    }
    
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {output_dir}/results.json")
    
    # ── Figures ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Per-profession ratios
    ax = axes[0, 0]
    profs_sorted = sorted(prof_ratios.items(), key=lambda x: x[1], reverse=True)
    names_p = [p[0] for p in profs_sorted]
    vals_p = [p[1] for p in profs_sorted]
    colors = ["#e74c3c" if v > 1.0 else "#3498db" for v in vals_p]
    ax.barh(range(len(names_p)), vals_p, color=colors, height=0.7)
    ax.set_yticks(range(len(names_p)))
    ax.set_yticklabels(names_p, fontsize=7)
    ax.axvline(x=1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Func/Cont TCD Ratio")
    ax.set_title(f"Per-Profession Ratio ({n_above_1}/{len(PROFESSIONS)} > 1.0)")
    ax.invert_yaxis()
    
    # 2. Race signed gaps
    ax = axes[0, 1]
    race_gaps = defaultdict(list)
    for r in rows:
        if r["race_a"] != r["race_b"]:
            key = f"{r['race_a']}-{r['race_b']}"
            race_gaps[key].append(r["signed_gap"])
    gap_means = [(k, np.mean(v)) for k, v in race_gaps.items()]
    gap_means.sort(key=lambda x: x[1])
    ax.barh([g[0] for g in gap_means], [g[1] for g in gap_means],
            color=["#e74c3c" if g[1] > 0 else "#3498db" for g in gap_means])
    ax.axvline(x=0, color="black", linestyle="--")
    ax.set_xlabel("Mean Signed Score Gap")
    ax.set_title("Directional Bias by Race Pair")
    
    # 3. Template ratios
    ax = axes[1, 0]
    tmpl_ratios = []
    for t_idx in range(len(TEMPLATES)):
        trows = [r for r in rows if r["template_id"] == t_idx]
        tf = np.mean([r["func_tcd"] for r in trows])
        tc = np.mean([r["cont_tcd"] for r in trows])
        tmpl_ratios.append(tf/tc if tc > 0 else 0)
    ax.bar(range(len(TEMPLATES)), tmpl_ratios, color="#2ecc71")
    ax.axhline(y=1.0, color="black", linestyle="--")
    ax.set_xlabel("Template")
    ax.set_ylabel("Func/Cont Ratio")
    ax.set_title("Ratio by Template")
    ax.set_xticks(range(len(TEMPLATES)))
    ax.set_xticklabels([f"T{i}" for i in range(len(TEMPLATES))], fontsize=8)
    
    # 4. Score gap distribution
    ax = axes[1, 1]
    ax.hist(score_gaps, bins=50, color="#9b59b6", alpha=0.8, edgecolor="white")
    ax.axvline(x=np.mean(score_gaps), color="red", linewidth=2, label=f"Mean={np.mean(score_gaps):.3f}")
    ax.set_xlabel("|Score Gap|")
    ax.set_ylabel("Count")
    ax.set_title("Score Gap Distribution")
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "scaled_results.pdf"), dpi=150, bbox_inches="tight")
    print(f"✅ Figure saved to {output_dir}/scaled_results.pdf")
    
    print(f"\n{'='*70}")
    print("✅ ALL DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/30_scaled")
    parser.add_argument("--all-queries", action="store_true", help="Run all 5 query formats (5x slower)")
    args = parser.parse_args()
    run_scaled(args.output_dir, use_all_queries=args.all_queries)
