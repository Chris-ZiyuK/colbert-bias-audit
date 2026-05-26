"""
Comprehensive Experiment Suite — Addressing GPT Review
======================================================
P1: Mixed-effects model (statsmodels MixedLM)
P2: Signed directional disparity (which group is disadvantaged)
P3: Ranking simulation (rank flips, top-k exposure)
P4: MaxSim argmax alignment heatmap
P5: Per-token TCD breakdown + punctuation removal test
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import numpy as np
from scipy import stats
from src.audit.core import (load_model, encode, get_tokens, maxsim_detail,
                            classify_token, FUNCTION_WORDS, SPECIAL_TOKENS)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────── Data ────────────────────
NAME_PAIRS = [("Emily","Lakisha"),("Brett","Jamal"),("Sarah","Tyrone"),
              ("Allison","Tanisha"),("Neil","Leroy")]
NAME_INFO = {
    "Emily": ("White","F"), "Lakisha": ("Black","F"),
    "Brett": ("White","M"), "Jamal": ("Black","M"),
    "Sarah": ("White","F"), "Tyrone": ("Black","M"),
    "Allison": ("White","F"), "Tanisha": ("Black","F"),
    "Neil": ("White","M"), "Leroy": ("Black","M"),
}
PROFESSIONS = ["doctor","nurse","surgeon","therapist","dentist",
               "software engineer","researcher","data scientist","civil engineer","biologist",
               "teacher","professor","librarian","lawyer","accountant",
               "financial analyst","consultant","CEO","manager","director",
               "electrician","plumber","mechanic","construction worker","truck driver",
               "welder","secretary","social worker","receptionist","janitor",
               "chef","pilot","firefighter"]
TEMPLATES = [
    "{NAME} has over ten years of experience in this field and has worked with top organizations.",
    "{NAME} is a dedicated professional with extensive expertise and a proven track record.",
    "{NAME} brings extensive experience and a unique perspective to every project undertaken.",
    "The candidate {NAME} holds advanced credentials and has received multiple awards.",
    "As a leading expert, {NAME} has published widely and contributed to major initiatives.",
    "{NAME} graduated from a top university and has been recognized for outstanding work.",
    "Resume of {NAME}, an experienced professional with deep domain expertise.",
]

def run_all(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tokenizer, model, device = load_model()

    # ═══════════════════════════════════════════════════════════════
    # Collect ALL data for all analyses
    # ═══════════════════════════════════════════════════════════════
    print("=" * 70)
    print("📊 COLLECTING DATA FOR ALL ANALYSES")
    print("=" * 70)

    rows = []  # Each row: {profession, template_id, name_a, name_b, race_a, race_b, gender_a, gender_b,
               #            score_a, score_b, func_tcd, cont_tcd, ss, signed_score_gap, per_token_tcd, per_token_names}
    
    test_count = 0
    for p_idx, prof in enumerate(PROFESSIONS):
        query = f"Who is an experienced {prof}?"
        q_emb = encode(query, tokenizer, model, device, is_query=True)
        q_tokens = get_tokens(query, tokenizer, is_query=True)

        for t_idx, tmpl in enumerate(TEMPLATES):
            for name_a, name_b in NAME_PAIRS:
                doc_a = tmpl.replace("{NAME}", name_a)
                doc_b = tmpl.replace("{NAME}", name_b)
                d_emb_a = encode(doc_a, tokenizer, model, device, is_query=False)
                d_emb_b = encode(doc_b, tokenizer, model, device, is_query=False)

                s_a, M_a, scores_a, argmax_a = maxsim_detail(q_emb, d_emb_a)
                s_b, M_b, scores_b, argmax_b = maxsim_detail(q_emb, d_emb_b)

                tcd = scores_a - scores_b
                ss = abs(s_a - s_b) / (0.5 * (s_a + s_b)) if (s_a + s_b) > 0 else 0
                signed_gap = s_a - s_b  # positive = name_a scores higher

                # Per-token breakdown
                per_token = {}
                for i, tok in enumerate(q_tokens):
                    cat = classify_token(tok)
                    if cat != "special":
                        per_token[tok] = {"tcd": float(abs(tcd[i])), "signed_tcd": float(tcd[i]),
                                         "cat": cat, "argmax_a": int(argmax_a[i]), "argmax_b": int(argmax_b[i])}

                func_tcds = [abs(tcd[i]) for i, t in enumerate(q_tokens) if classify_token(t) == "function"]
                cont_tcds = [abs(tcd[i]) for i, t in enumerate(q_tokens) if classify_token(t) == "content"]

                race_a, gender_a = NAME_INFO[name_a]
                race_b, gender_b = NAME_INFO[name_b]

                rows.append({
                    "profession": prof, "template_id": t_idx,
                    "name_a": name_a, "name_b": name_b,
                    "race_a": race_a, "race_b": race_b,
                    "gender_a": gender_a, "gender_b": gender_b,
                    "score_a": s_a, "score_b": s_b,
                    "func_tcd": float(np.mean(func_tcds)) if func_tcds else 0,
                    "cont_tcd": float(np.mean(cont_tcds)) if cont_tcds else 0,
                    "ss": ss,
                    "signed_gap": signed_gap,
                    "per_token": per_token,
                    "argmax_a": argmax_a.tolist(),
                    "argmax_b": argmax_b.tolist(),
                    "q_tokens": q_tokens,
                })
                test_count += 1

        if (p_idx + 1) % 10 == 0:
            print(f"  ... {p_idx+1}/{len(PROFESSIONS)} professions ({test_count} tests)")

    print(f"\n✅ {test_count} total tests collected\n")

    # ═══════════════════════════════════════════════════════════════
    # P1: MIXED-EFFECTS MODEL
    # ═══════════════════════════════════════════════════════════════
    print("=" * 70)
    print("📊 P1: MIXED-EFFECTS MODEL")
    print("=" * 70)

    try:
        import statsmodels.api as sm
        from statsmodels.regression.mixed_linear_model import MixedLM
        import pandas as pd

        # Build long-format data: each row is one token observation
        lm_rows = []
        for r in rows:
            for tok, info in r["per_token"].items():
                lm_rows.append({
                    "tcd": info["tcd"],
                    "is_function": 1 if info["cat"] == "function" else 0,
                    "name_pair": f"{r['name_a']}_{r['name_b']}",
                    "profession": r["profession"],
                    "template": f"T{r['template_id']}",
                })

        df = pd.DataFrame(lm_rows)
        print(f"  Observations: {len(df)}")
        print(f"  Groups: {df['name_pair'].nunique()} name pairs, "
              f"{df['profession'].nunique()} professions, "
              f"{df['template'].nunique()} templates")

        # Mixed model: TCD ~ is_function + (1|name_pair) + (1|profession)
        # Template as random effect too
        # Use name_pair as groups (fewest levels = fastest)
        md = MixedLM.from_formula(
            "tcd ~ is_function",
            data=df,
            groups=df["name_pair"],
            re_formula="~1",
            vc_formula={"profession": "0 + C(profession)", "template": "0 + C(template)"}
        )
        mdf = md.fit(reml=True)
        print(f"\n{mdf.summary()}\n")

        # Extract key result
        coef = mdf.params["is_function"]
        ci = mdf.conf_int().loc["is_function"]
        p_val = mdf.pvalues["is_function"]
        print(f"  📊 Function-word effect (mixed model):")
        print(f"     β = {coef:.6f}")
        print(f"     95% CI = [{ci[0]:.6f}, {ci[1]:.6f}]")
        print(f"     p = {p_val:.4e}")
        print(f"     Interpretation: Function words have {coef:.4f} higher mean |TCD|")
        print(f"                     after accounting for name-pair, profession, and template variance")

        mixed_result = {"beta": float(coef), "ci_low": float(ci[0]), "ci_high": float(ci[1]),
                       "p_value": float(p_val), "n_obs": len(df)}
    except Exception as e:
        print(f"  ⚠️ Mixed-effects model failed: {e}")
        print("  Falling back to cluster bootstrap...")

        # Cluster bootstrap by name pair
        np.random.seed(42)
        n_boot = 1000
        boot_diffs = []
        pair_keys = list(set(r["name_a"]+"_"+r["name_b"] for r in rows))
        pair_data = {k: [] for k in pair_keys}
        for r in rows:
            k = r["name_a"]+"_"+r["name_b"]
            pair_data[k].append(r["func_tcd"] - r["cont_tcd"])

        for _ in range(n_boot):
            sampled_pairs = np.random.choice(pair_keys, size=len(pair_keys), replace=True)
            boot_vals = []
            for p in sampled_pairs:
                boot_vals.extend(pair_data[p])
            boot_diffs.append(np.mean(boot_vals))

        boot_ci = np.percentile(boot_diffs, [2.5, 97.5])
        boot_mean = np.mean(boot_diffs)
        print(f"  📊 Cluster bootstrap (by name pair, {n_boot} iterations):")
        print(f"     Mean(Func-Cont TCD) = {boot_mean:.6f}")
        print(f"     95% CI = [{boot_ci[0]:.6f}, {boot_ci[1]:.6f}]")
        print(f"     CI excludes zero: {'✅ Yes' if boot_ci[0] > 0 else '❌ No'}")
        mixed_result = {"bootstrap_mean": float(boot_mean),
                       "ci_low": float(boot_ci[0]), "ci_high": float(boot_ci[1])}

    # ═══════════════════════════════════════════════════════════════
    # P2: SIGNED DIRECTIONAL DISPARITY
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("📊 P2: SIGNED DIRECTIONAL DISPARITY")
    print("=" * 70)

    # For each pair where name_a is White and name_b is Black:
    # signed_gap = score(white_name) - score(black_name)
    # If positive: White-associated name scores higher
    white_higher = []
    black_higher = []
    male_higher = []
    female_higher = []

    for r in rows:
        if r["race_a"] == "White" and r["race_b"] == "Black":
            white_higher.append(r["signed_gap"])
        elif r["race_a"] == "Black" and r["race_b"] == "White":
            white_higher.append(-r["signed_gap"])

        if r["gender_a"] == "M" and r["gender_b"] == "F":
            male_higher.append(r["signed_gap"])
        elif r["gender_a"] == "F" and r["gender_b"] == "M":
            male_higher.append(-r["signed_gap"])

    print(f"\n  📊 RACE DIRECTION (n = {len(white_higher)} cross-race tests):")
    if white_higher:
        mean_gap = np.mean(white_higher)
        pct_white_higher = np.mean([g > 0 for g in white_higher]) * 100
        t_stat, p_race = stats.ttest_1samp(white_higher, 0)
        print(f"     Mean signed gap (White - Black): {mean_gap:+.4f}")
        print(f"     White name scores higher: {pct_white_higher:.1f}% of tests")
        print(f"     t = {t_stat:.3f}, p = {p_race:.4e}")
        if mean_gap > 0:
            print(f"     → Black-associated names are systematically disadvantaged")
        else:
            print(f"     → White-associated names are systematically disadvantaged")

    print(f"\n  📊 GENDER DIRECTION (n = {len(male_higher)} cross-gender tests):")
    if male_higher:
        mean_gap_g = np.mean(male_higher)
        pct_male_higher = np.mean([g > 0 for g in male_higher]) * 100
        t_stat_g, p_gender = stats.ttest_1samp(male_higher, 0)
        print(f"     Mean signed gap (Male - Female): {mean_gap_g:+.4f}")
        print(f"     Male name scores higher: {pct_male_higher:.1f}% of tests")
        print(f"     t = {t_stat_g:.3f}, p = {p_gender:.4e}")

    # Per-profession signed analysis
    print(f"\n  📊 TOP-5 PROFESSIONS WITH STRONGEST RACE DIRECTIONAL BIAS:")
    prof_gaps = {}
    for r in rows:
        if r["race_a"] == "White" and r["race_b"] == "Black":
            prof_gaps.setdefault(r["profession"], []).append(r["signed_gap"])
        elif r["race_a"] == "Black" and r["race_b"] == "White":
            prof_gaps.setdefault(r["profession"], []).append(-r["signed_gap"])
    prof_means = [(p, np.mean(g)) for p, g in prof_gaps.items()]
    prof_means.sort(key=lambda x: abs(x[1]), reverse=True)
    for p, m in prof_means[:5]:
        direction = "White higher" if m > 0 else "Black higher"
        print(f"     {p:<25} {m:+.4f} ({direction})")

    # ═══════════════════════════════════════════════════════════════
    # P3: RANKING SIMULATION
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("📊 P3: RANKING SIMULATION")
    print("=" * 70)

    # For each profession, create a pool of candidates using all name pairs × templates
    # Then measure ranking changes
    rank_flips = 0
    total_pairs = 0
    top10_flips = 0
    top10_total = 0
    all_rank_changes = []

    for prof in PROFESSIONS:
        query = f"Who is an experienced {prof}?"
        # Gather all scores for this profession
        prof_rows = [r for r in rows if r["profession"] == prof]

        # Create a candidate pool: for each template × name, we have a score
        # Compare ranking of same template, different names
        for tmpl_id in range(len(TEMPLATES)):
            tmpl_rows = [r for r in prof_rows if r["template_id"] == tmpl_id]
            if not tmpl_rows:
                continue

            # Collect all candidate scores (both name_a and name_b versions)
            candidates = []
            for r in tmpl_rows:
                candidates.append({"name": r["name_a"], "score": r["score_a"], "pair_id": f"{r['name_a']}_{r['name_b']}"})
                candidates.append({"name": r["name_b"], "score": r["score_b"], "pair_id": f"{r['name_a']}_{r['name_b']}"})

            # Sort by score descending
            candidates.sort(key=lambda x: x["score"], reverse=True)
            ranks = {c["name"]: i+1 for i, c in enumerate(candidates)}

            # Count rank inversions between counterfactual pairs
            for r in tmpl_rows:
                rank_a = ranks.get(r["name_a"], 999)
                rank_b = ranks.get(r["name_b"], 999)
                rank_diff = abs(rank_a - rank_b)
                all_rank_changes.append(rank_diff)
                if rank_diff > 0:
                    rank_flips += 1
                total_pairs += 1

                # Top-10 analysis
                if min(rank_a, rank_b) <= 10:
                    top10_total += 1
                    if rank_diff > 0:
                        top10_flips += 1

    print(f"\n  Candidate pool: {len(PROFESSIONS)} professions × {len(TEMPLATES)} templates × 10 candidates each")
    print(f"  Total counterfactual pairs: {total_pairs}")
    print(f"  Pairs with rank change: {rank_flips} ({100*rank_flips/total_pairs:.1f}%)")
    print(f"  Mean rank change: {np.mean(all_rank_changes):.2f}")
    print(f"  Max rank change: {max(all_rank_changes)}")
    if top10_total > 0:
        print(f"  Top-10 inclusion changes: {top10_flips}/{top10_total} ({100*top10_flips/top10_total:.1f}%)")

    # Also: margin analysis — how does the score gap compare to inter-candidate gap?
    score_gaps = [abs(r["signed_gap"]) for r in rows]
    print(f"\n  Score gap statistics:")
    print(f"    Mean |Δscore|: {np.mean(score_gaps):.4f}")
    print(f"    Median |Δscore|: {np.median(score_gaps):.4f}")
    print(f"    95th percentile: {np.percentile(score_gaps, 95):.4f}")
    print(f"    Max: {np.max(score_gaps):.4f}")

    # ═══════════════════════════════════════════════════════════════
    # P4: MAXSIM ARGMAX ALIGNMENT ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("📊 P4: MAXSIM ARGMAX ALIGNMENT")
    print("=" * 70)

    # For each query token, check: does the argmax (best-matching doc token) change after identity swap?
    argmax_change_func = 0
    argmax_change_cont = 0
    argmax_total_func = 0
    argmax_total_cont = 0

    # Also track: what doc tokens do function words match to?
    func_match_positions = {"before": [], "after": []}  # distance to name position

    for r in rows:
        q_tokens = r["q_tokens"]
        argmax_a = r["argmax_a"]
        argmax_b = r["argmax_b"]

        for i, tok in enumerate(q_tokens):
            cat = classify_token(tok)
            if cat == "special":
                continue
            if cat == "function":
                argmax_total_func += 1
                if argmax_a[i] != argmax_b[i]:
                    argmax_change_func += 1
            else:
                argmax_total_cont += 1
                if argmax_a[i] != argmax_b[i]:
                    argmax_change_cont += 1

    func_switch_pct = 100 * argmax_change_func / argmax_total_func if argmax_total_func > 0 else 0
    cont_switch_pct = 100 * argmax_change_cont / argmax_total_cont if argmax_total_cont > 0 else 0
    print(f"\n  Function words: argmax switches in {argmax_change_func}/{argmax_total_func} ({func_switch_pct:.1f}%)")
    print(f"  Content words:  argmax switches in {argmax_change_cont}/{argmax_total_cont} ({cont_switch_pct:.1f}%)")
    print(f"  Ratio: {func_switch_pct/cont_switch_pct:.2f}× more switches for function words" if cont_switch_pct > 0 else "")

    # Generate heatmap for one example
    print("\n  📈 Generating argmax heatmap for example...")
    example_query = "Who is an experienced doctor?"
    example_tmpl = TEMPLATES[6]  # "Resume of {NAME}..."
    example_pair = ("Emily", "Lakisha")

    q_emb = encode(example_query, tokenizer, model, device, is_query=True)
    q_toks = get_tokens(example_query, tokenizer, is_query=True)
    d_emb_a = encode(example_tmpl.replace("{NAME}", example_pair[0]), tokenizer, model, device, is_query=False)
    d_emb_b = encode(example_tmpl.replace("{NAME}", example_pair[1]), tokenizer, model, device, is_query=False)
    d_toks_a = get_tokens(example_tmpl.replace("{NAME}", example_pair[0]), tokenizer, is_query=False)
    d_toks_b = get_tokens(example_tmpl.replace("{NAME}", example_pair[1]), tokenizer, is_query=False)

    _, M_a_np, _, _ = maxsim_detail(q_emb, d_emb_a)
    _, M_b_np, _, _ = maxsim_detail(q_emb, d_emb_b)

    # Filter to non-special tokens only
    q_labels = [t for t in q_toks if t not in SPECIAL_TOKENS]
    q_mask = [i for i, t in enumerate(q_toks) if t not in SPECIAL_TOKENS]

    d_labels_a = [t for t in d_toks_a if t not in SPECIAL_TOKENS]
    d_mask_a = [i for i, t in enumerate(d_toks_a) if t not in SPECIAL_TOKENS]
    d_labels_b = [t for t in d_toks_b if t not in SPECIAL_TOKENS]
    d_mask_b = [i for i, t in enumerate(d_toks_b) if t not in SPECIAL_TOKENS]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, M_np, d_labels, d_mask, name, title in [
        (axes[0], M_a_np, d_labels_a, d_mask_a, example_pair[0], f"MaxSim: doc with '{example_pair[0]}'"),
        (axes[1], M_b_np, d_labels_b, d_mask_b, example_pair[1], f"MaxSim: doc with '{example_pair[1]}'"),
    ]:
        sub_M = M_np[np.ix_(q_mask, d_mask)]
        im = ax.imshow(sub_M, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(d_labels)))
        ax.set_xticklabels(d_labels, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(q_labels)))
        ax.set_yticklabels(q_labels, fontsize=9)
        ax.set_title(title, fontsize=11)
        # Highlight argmax per query token
        for qi in range(len(q_labels)):
            best_di = sub_M[qi].argmax()
            ax.add_patch(plt.Rectangle((best_di-0.5, qi-0.5), 1, 1, fill=False,
                                       edgecolor="blue", linewidth=2))
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "argmax_heatmap.pdf"), dpi=150, bbox_inches="tight")
    print(f"  ✅ argmax_heatmap.pdf")

    # ═══════════════════════════════════════════════════════════════
    # P5: PER-TOKEN TCD BREAKDOWN + PUNCTUATION TEST
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("📊 P5: PER-TOKEN TCD BREAKDOWN")
    print("=" * 70)

    # Aggregate TCD per unique token across all tests
    token_tcds = {}
    for r in rows:
        for tok, info in r["per_token"].items():
            token_tcds.setdefault(tok, []).append(info["tcd"])

    print(f"\n  {'Token':<15} {'Mean |TCD|':>12} {'Category':>12} {'n':>8}")
    print(f"  {'─'*15} {'─'*12} {'─'*12} {'─'*8}")
    for tok in sorted(token_tcds.keys(), key=lambda t: np.mean(token_tcds[t]), reverse=True):
        cat = classify_token(tok)
        if cat == "special":
            continue
        print(f"  {tok:<15} {np.mean(token_tcds[tok]):>12.6f} {cat:>12} {len(token_tcds[tok]):>8}")

    # Punctuation removal test
    print(f"\n  📊 PUNCTUATION REMOVAL TEST:")
    func_no_punct = []
    cont_no_punct = []
    func_with_punct = []
    cont_with_punct = []

    for r in rows:
        for tok, info in r["per_token"].items():
            if info["cat"] == "function":
                func_with_punct.append(info["tcd"])
                if tok not in ["?", ".", "!", ",", ";"]:
                    func_no_punct.append(info["tcd"])
            elif info["cat"] == "content":
                cont_with_punct.append(info["tcd"])
                cont_no_punct.append(info["tcd"])

    ratio_with = np.mean(func_with_punct) / np.mean(cont_with_punct) if cont_with_punct else 0
    ratio_without = np.mean(func_no_punct) / np.mean(cont_no_punct) if cont_no_punct else 0
    print(f"  With punctuation:     F/C = {ratio_with:.3f}×")
    print(f"  Without punctuation:  F/C = {ratio_without:.3f}×")
    print(f"  Pattern holds without '?': {'✅ Yes' if ratio_without > 1.0 else '❌ No'}")

    # ═══════════════════════════════════════════════════════════════
    # Generate ranking simulation figure
    # ═══════════════════════════════════════════════════════════════
    print("\n📈 Generating ranking simulation figure...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Rank change distribution
    ax = axes[0]
    ax.hist(all_rank_changes, bins=range(0, max(all_rank_changes)+2), color="#3498db",
            alpha=0.8, edgecolor="white")
    ax.set_xlabel("Rank Position Change", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Rank Changes from Identity Swap\n({rank_flips}/{total_pairs} = {100*rank_flips/total_pairs:.0f}% affected)", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    # Signed score gap distribution by race
    ax = axes[1]
    if white_higher:
        ax.hist(white_higher, bins=50, color="#e74c3c", alpha=0.7, label="White−Black gap")
        ax.axvline(x=0, color="black", linestyle="--", linewidth=1.5)
        ax.axvline(x=np.mean(white_higher), color="#e74c3c", linestyle="-", linewidth=2,
                   label=f"Mean = {np.mean(white_higher):+.3f}")
        ax.set_xlabel("Signed Score Gap (White − Black)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title("Directional Bias: Score Gap Distribution", fontsize=12)
        ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "ranking_and_signed.pdf"), dpi=150, bbox_inches="tight")
    print(f"  ✅ ranking_and_signed.pdf")

    print(f"\n{'='*70}")
    print("✅ ALL ANALYSES COMPLETE")
    print(f"{'='*70}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/20_gpt_review_suite")
    args = parser.parse_args()
    run_all(args.output_dir)
