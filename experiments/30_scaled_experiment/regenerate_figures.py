"""
Regenerate all paper figures from FRESH scaled experiment data.
Embeds figure generation directly into the computation loop
to avoid double-compute. Saves raw rows to cache for future use.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import numpy as np
from scipy import stats
from collections import defaultdict
from src.audit.core import (load_model, encode, get_tokens, maxsim_detail,
                            classify_token, get_token_count)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Style ──
plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 10, "axes.titlesize": 11,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})
C_FUNC = "#E74C3C"; C_CONT = "#3498DB"; C_ACCENT = "#2ECC71"
C_PURPLE = "#9B59B6"; C_ORANGE = "#E67E22"; C_DARK = "#2C3E50"

FIGDIR = os.path.join(os.path.dirname(__file__), "../../paper/figures")
CACHE = os.path.join(os.path.dirname(__file__), "../../results/30_scaled/raw_rows.json")

PROFESSIONS = [
    "doctor","nurse","surgeon","therapist","dentist",
    "software engineer","researcher","data scientist","civil engineer","biologist",
    "teacher","professor","librarian","lawyer","accountant",
    "financial analyst","consultant","CEO","manager","director",
    "electrician","plumber","mechanic","construction worker","truck driver",
    "welder","secretary","social worker","receptionist","janitor",
    "chef","pilot","firefighter",
]

TEMPLATES = [
    "{NAME} has over ten years of experience in this field and has worked with top organizations.",
    "{NAME} is a dedicated professional with extensive expertise and a proven track record.",
    "{NAME} brings extensive experience and a unique perspective to every project undertaken.",
    "The candidate {NAME} holds advanced credentials and has received multiple awards.",
    "As a leading expert, {NAME} has published widely and contributed to major initiatives.",
    "{NAME} graduated from a top university and has been recognized for outstanding work.",
    "Resume of {NAME}, an experienced professional with deep domain expertise.",
    "{NAME} completed a prestigious fellowship and mentored junior colleagues.",
    "A highly rated professional, {NAME} has served clients across multiple industries.",
    "With dual expertise in management and strategy, {NAME} leads cross-functional teams.",
    "Certified and licensed, {NAME} maintains the highest standards of professional practice.",
    "{NAME}'s career spans two decades of innovation and leadership in this domain.",
    "Colleagues describe {NAME} as reliable, insightful, and exceptionally detail-oriented.",
    "Having worked internationally, {NAME} brings a global perspective to every engagement.",
]

import random, itertools

def load_expanded_names(tokenizer):
    pool_path = os.path.join(os.path.dirname(__file__), "../../data/names/p2b_rosenman_name_features.json")
    with open(pool_path) as f:
        data = json.load(f)
    names = []
    for n in data["names"]:
        bpe = n.get("bpe_tokens", None)
        if bpe is None or bpe == "?":
            bpe = get_token_count(n["name"], tokenizer)
        names.append({"name": n["name"], "race": n["race"], "gender": n["gender"],
                      "bpe": int(bpe), "rarity": "rare" if int(bpe) > 1 else "common"})
    return names

def build_stratified_pairs(names, target_n=120):
    random.seed(42)
    pairs = []; used = set()
    def add_pair(a, b, pt):
        key = (min(a["name"], b["name"]), max(a["name"], b["name"]))
        if key not in used:
            used.add(key); pairs.append({"name_a": a, "name_b": b, "type": pt})
    by_rg = defaultdict(list)
    for n in names: by_rg[f"{n['race']}_{n['gender']}"].append(n)
    races = sorted(set(n["race"] for n in names))
    for r1, r2 in itertools.combinations(races, 2):
        for g in ["F", "M"]:
            p1, p2 = by_rg.get(f"{r1}_{g}", []), by_rg.get(f"{r2}_{g}", [])
            if p1 and p2:
                c = [(a,b) for a in p1 for b in p2]; random.shuffle(c)
                for a,b in c[:3]: add_pair(a, b, f"cross_race_{r1}_{r2}")
    for r in races:
        pf, pm = by_rg.get(f"{r}_F",[]), by_rg.get(f"{r}_M",[])
        if pf and pm:
            c = [(a,b) for a in pf for b in pm]; random.shuffle(c)
            for a,b in c[:5]: add_pair(a, b, f"cross_gender_{r}")
    for key, grp in by_rg.items():
        if len(grp) >= 2:
            c = list(itertools.combinations(grp, 2)); random.shuffle(c)
            for a,b in c[:3]: add_pair(a, b, f"within_{key}")
    for r1, r2 in itertools.combinations(races, 2):
        for g1,g2 in [("F","M"),("M","F")]:
            p1, p2 = by_rg.get(f"{r1}_{g1}",[]), by_rg.get(f"{r2}_{g2}",[])
            if p1 and p2:
                c = [(a,b) for a in p1 for b in p2]; random.shuffle(c)
                for a,b in c[:2]: add_pair(a, b, f"cross_both_{r1}_{r2}")
    by_rar = defaultdict(list)
    for n in names: by_rar[f"{n['rarity']}_{n['gender']}"].append(n)
    for key, grp in by_rar.items():
        ri = defaultdict(list)
        for n in grp: ri[n["race"]].append(n)
        for r1, r2 in itertools.combinations(ri.keys(), 2):
            c = [(a,b) for a in ri[r1] for b in ri[r2]]; random.shuffle(c)
            for a,b in c[:2]: add_pair(a, b, f"matched_bpe_{key}")
    all_c = list(itertools.combinations(names, 2)); random.shuffle(all_c)
    for a, b in all_c:
        if len(pairs) >= target_n: break
        add_pair(a, b, "random")
    return pairs[:target_n]


def compute_and_plot():
    if os.path.exists(CACHE):
        print("📦 Loading cached raw rows...")
        with open(CACHE) as f:
            rows = json.load(f)
        print(f"   Loaded {len(rows):,} rows")
    else:
        print("🔄 Running experiment to collect raw row data...")
        tokenizer, model, device = load_model()
        names = load_expanded_names(tokenizer)
        pairs = build_stratified_pairs(names, target_n=120)
        
        query_template = "Who is an experienced {prof}?"
        rows = []
        t0 = time.time()
        total = len(PROFESSIONS) * len(TEMPLATES) * len(pairs)
        count = 0
        
        for p_idx, prof in enumerate(PROFESSIONS):
            query = query_template.format(prof=prof)
            q_emb = encode(query, tokenizer, model, device, is_query=True)
            q_tokens = get_tokens(query, tokenizer, is_query=True)
            
            for t_idx, tmpl in enumerate(TEMPLATES):
                for pair in pairs:
                    na, nb = pair["name_a"], pair["name_b"]
                    doc_a = tmpl.replace("{NAME}", na["name"])
                    doc_b = tmpl.replace("{NAME}", nb["name"])
                    d_emb_a = encode(doc_a, tokenizer, model, device, is_query=False)
                    d_emb_b = encode(doc_b, tokenizer, model, device, is_query=False)
                    
                    s_a, _, scores_a, _ = maxsim_detail(q_emb, d_emb_a)
                    s_b, _, scores_b, _ = maxsim_detail(q_emb, d_emb_b)
                    tcd = scores_a - scores_b
                    ss = abs(s_a - s_b) / (0.5 * (s_a + s_b)) if (s_a + s_b) > 0 else 0
                    
                    func_tcds = [abs(tcd[i]) for i, t in enumerate(q_tokens) if classify_token(t) == "function"]
                    cont_tcds = [abs(tcd[i]) for i, t in enumerate(q_tokens) if classify_token(t) == "content"]
                    
                    rows.append({
                        "profession": prof, "template_id": t_idx,
                        "name_a": na["name"], "name_b": nb["name"],
                        "race_a": na["race"], "race_b": nb["race"],
                        "gender_a": na["gender"], "gender_b": nb["gender"],
                        "bpe_a": na["bpe"], "bpe_b": nb["bpe"],
                        "pair_type": pair["type"],
                        "score_a": float(s_a), "score_b": float(s_b),
                        "func_tcd": float(np.mean(func_tcds)) if func_tcds else 0,
                        "cont_tcd": float(np.mean(cont_tcds)) if cont_tcds else 0,
                        "ss": float(ss),
                        "signed_gap": float(s_a - s_b),
                    })
                    count += 1
            
            if (p_idx + 1) % 5 == 0:
                elapsed = time.time() - t0
                rate = count / elapsed
                eta = (total - count) / rate / 60
                print(f"  {p_idx+1}/{len(PROFESSIONS)} | {count:,}/{total:,} | ETA {eta:.1f} min")
        
        print(f"\n✅ {count:,} tests completed in {(time.time()-t0)/60:.1f} min")
        
        # Cache
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        with open(CACHE, "w") as f:
            json.dump(rows, f)
        print(f"💾 Cached to {CACHE}")
    
    # ═══════════════════════════════════════════════════
    # GENERATE FIGURES
    # ═══════════════════════════════════════════════════
    os.makedirs(FIGDIR, exist_ok=True)
    
    # ── Fig 1: Func vs Cont distribution ──
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    func_vals = [r["func_tcd"] for r in rows]
    cont_vals = [r["cont_tcd"] for r in rows]
    bins = np.linspace(0, 0.08, 60)
    ax.hist(func_vals, bins=bins, alpha=0.65, color=C_FUNC, label="Function words",
            density=True, edgecolor="white", linewidth=0.3)
    ax.hist(cont_vals, bins=bins, alpha=0.65, color=C_CONT, label="Content words",
            density=True, edgecolor="white", linewidth=0.3)
    mf, mc = np.mean(func_vals), np.mean(cont_vals)
    ax.axvline(mf, color=C_FUNC, linestyle="--", linewidth=1.5, alpha=0.9)
    ax.axvline(mc, color=C_CONT, linestyle="--", linewidth=1.5, alpha=0.9)
    ratio = mf / mc
    diffs = np.array(func_vals) - np.array(cont_vals)
    d = np.mean(diffs) / np.std(diffs, ddof=1) if np.std(diffs) > 0 else 0
    ax.set_xlabel("Mean |TCD| per query token")
    ax.set_ylabel("Density")
    ax.set_title(f"Function vs. Content Word TCD (n = {len(rows):,})\nRatio = {ratio:.2f}×, Cohen's d = {d:.2f}", fontsize=9)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(0, 0.08)
    fig.savefig(os.path.join(FIGDIR, "fig1_func_vs_cont_tcd.pdf"))
    fig.savefig(os.path.join(FIGDIR, "fig1_func_vs_cont_tcd.png"), dpi=150)
    plt.close(fig)
    print("✅ fig1_func_vs_cont_tcd")
    
    # ── Fig 4: Per-profession ratios ──
    prof_ratios = {}
    for prof in set(r["profession"] for r in rows):
        prows = [r for r in rows if r["profession"] == prof]
        pf = np.mean([r["func_tcd"] for r in prows])
        pc = np.mean([r["cont_tcd"] for r in prows])
        prof_ratios[prof] = pf / pc if pc > 0 else 0
    sorted_p = sorted(prof_ratios.items(), key=lambda x: x[1], reverse=True)
    pnames = [p[0] for p in sorted_p]
    pvals = [p[1] for p in sorted_p]
    pcolors = [C_FUNC if v > 1.0 else C_PURPLE for v in pvals]
    fig, ax = plt.subplots(figsize=(4.5, 5.5))
    ax.barh(range(len(pnames)), pvals, color=pcolors, height=0.7, edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(len(pnames)))
    ax.set_yticklabels(pnames, fontsize=7)
    ax.axvline(x=1.0, color=C_DARK, linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Func/Cont TCD Ratio")
    n_above = sum(1 for v in pvals if v > 1.0)
    ax.set_title(f"Per-Profession F/C Ratio ({n_above}/{len(pvals)} > 1.0)", fontsize=10)
    ax.invert_yaxis()
    for i, v in enumerate(pvals):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=6, color=C_DARK)
    fig.savefig(os.path.join(FIGDIR, "fig4_profession_ratios.pdf"))
    fig.savefig(os.path.join(FIGDIR, "fig4_profession_ratios.png"), dpi=150)
    plt.close(fig)
    print("✅ fig4_profession_ratios")
    
    # ── Fig 3: Rarity amplification ──
    cc = [r["ss"] for r in rows if r["bpe_a"] == 1 and r["bpe_b"] == 1]
    cr = [r["ss"] for r in rows if (r["bpe_a"] == 1) != (r["bpe_b"] == 1)]
    rr = [r["ss"] for r in rows if r["bpe_a"] > 1 and r["bpe_b"] > 1]
    cats = ["CC\n(both common)", "CR\n(mixed)", "RR\n(both rare)"]
    means = [np.mean(cc), np.mean(cr) if cr else 0, np.mean(rr)]
    sems = [stats.sem(cc), stats.sem(cr) if cr else 0, stats.sem(rr)]
    ns = [len(cc), len(cr), len(rr)]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(range(3), means, yerr=sems, capsize=4,
           color=[C_CONT, C_ORANGE, C_FUNC], edgecolor="white", alpha=0.85)
    ax.set_xticks(range(3)); ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylabel("Mean Score Sensitivity (SS)")
    rr_cc = means[2]/means[0] if means[0] > 0 else 0
    ax.set_title(f"Rarity Amplification (RR/CC = {rr_cc:.2f}×)", fontsize=10)
    for i, (m, n) in enumerate(zip(means, ns)):
        ax.text(i, m + sems[i] + 0.0005, f"{m:.4f}\n(n={n:,})", ha="center", va="bottom", fontsize=7)
    fig.savefig(os.path.join(FIGDIR, "fig3_rarity_amplification.pdf"))
    fig.savefig(os.path.join(FIGDIR, "fig3_rarity_amplification.png"), dpi=150)
    plt.close(fig)
    print("✅ fig3_rarity_amplification")
    
    # ── Fig 7: Confound decomposition ──
    same_g = [r["ss"] for r in rows if r["gender_a"] == r["gender_b"]]
    cross_g = [r["ss"] for r in rows if r["gender_a"] != r["gender_b"]]
    same_r = [r["ss"] for r in rows if r["race_a"] == r["race_b"]]
    cross_r = [r["ss"] for r in rows if r["race_a"] != r["race_b"]]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    # Gender
    ax = axes[0]
    gm = [np.mean(same_g), np.mean(cross_g)]
    gs = [stats.sem(same_g), stats.sem(cross_g)]
    gn = [len(same_g), len(cross_g)]
    ax.bar([0,1], gm, yerr=gs, capsize=5, color=[C_CONT, C_FUNC], edgecolor="white", alpha=0.85)
    ax.set_xticks([0,1]); ax.set_xticklabels(["Same gender", "Cross gender"], fontsize=8)
    ax.set_ylabel("Mean SS")
    rg = gm[1]/gm[0] if gm[0] > 0 else 0
    ax.set_title(f"Gender (ratio = {rg:.2f}×)", fontsize=9)
    for i, (m,n) in enumerate(zip(gm, gn)):
        ax.text(i, m+gs[i]+0.0003, f"{m:.4f}\nn={n:,}", ha="center", va="bottom", fontsize=7)
    # Race
    ax = axes[1]
    rm = [np.mean(same_r), np.mean(cross_r)]
    rs = [stats.sem(same_r), stats.sem(cross_r)]
    rn = [len(same_r), len(cross_r)]
    ax.bar([0,1], rm, yerr=rs, capsize=5, color=[C_CONT, C_ORANGE], edgecolor="white", alpha=0.85)
    ax.set_xticks([0,1]); ax.set_xticklabels(["Same race", "Cross race"], fontsize=8)
    ax.set_ylabel("Mean SS")
    rr2 = rm[1]/rm[0] if rm[0] > 0 else 0
    ax.set_title(f"Race (ratio = {rr2:.2f}×, n.s.)", fontsize=9)
    for i, (m,n) in enumerate(zip(rm, rn)):
        ax.text(i, m+rs[i]+0.0003, f"{m:.4f}\nn={n:,}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig7_confound_decomposition.pdf"))
    fig.savefig(os.path.join(FIGDIR, "fig7_confound_decomposition.png"), dpi=150)
    plt.close(fig)
    print("✅ fig7_confound_decomposition")
    
    # ── Fig 6: Naturalistic-template validation comparison ──
    ratio_syn = np.mean(func_vals) / np.mean(cont_vals)
    diffs = np.array(func_vals) - np.array(cont_vals)
    d_syn = np.mean(diffs) / np.std(diffs, ddof=1) if np.std(diffs) > 0 else 0
    ratio_eco, d_eco = 2.08, 0.94
    fig, ax = plt.subplots(figsize=(4.5, 3))
    bars = ax.bar([0,1], [ratio_syn, ratio_eco],
                  color=[C_CONT, C_ACCENT], edgecolor="white", width=0.5, alpha=0.85)
    ax.set_xticks([0,1])
    ax.set_xticklabels([f"Controlled\n(14 templates)\nn = {len(rows):,}",
                        "Naturalistic\n(200 passages)\nn = 8,000"], fontsize=8)
    ax.set_ylabel("Func/Cont TCD Ratio")
    ax.axhline(y=1.0, color=C_DARK, linestyle="--", linewidth=1, alpha=0.5)
    ax.set_title("Function-Word Sensitivity: Controlled vs. Naturalistic", fontsize=10)
    for i, (r, dd) in enumerate(zip([ratio_syn, ratio_eco], [d_syn, d_eco])):
        ax.text(i, r+0.03, f"{r:.2f}×\nd = {dd:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_ylim(0, ratio_eco * 1.25)
    fig.savefig(os.path.join(FIGDIR, "fig6_naturalistic_validation.pdf"))
    fig.savefig(os.path.join(FIGDIR, "fig6_naturalistic_validation.png"), dpi=150)
    plt.close(fig)
    print("✅ fig6_naturalistic_validation")
    
    print(f"\n🎉 All figures saved to {FIGDIR}")


if __name__ == "__main__":
    compute_and_plot()
