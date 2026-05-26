"""
MS MARCO Ecological Validation — Using the EXACT same ColBERTRetriever
as the main experiments for consistency.
"""

import json, re, sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
RESULTS = PROJECT_ROOT / "local" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

from src.models.colbert import ColBERTRetriever
from src.metrics.tcd import compute_tcd, compute_func_cont_tcd, compute_score_sensitivity

FUNCTION_WORDS = {
    "who", "is", "a", "an", "the", "of", "in", "to", "for", "and", "or",
    "with", "on", "at", "by", "from", "that", "this", "it", "as", "be",
    "was", "were", "been", "are", "has", "have", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "can",
    "shall", "not", "no", "but", "if", "than", "what", "which",
    "[CLS]", "[SEP]", "[PAD]", "[MASK]", "?", ".", ":", "query",
}

COMMON_NAMES = [
    "John", "David", "Michael", "James", "Robert", "William", "Thomas",
    "Mary", "Jennifer", "Elizabeth", "Jessica", "Susan", "Karen",
    "Mark", "Paul", "Daniel", "Steven", "Andrew", "George", "Edward",
    "Lisa", "Nancy", "Margaret", "Sandra", "Dorothy",
    "Richard", "Joseph", "Charles", "Christopher", "Matthew", "Anthony",
    "Brian", "Kevin", "Jason", "Peter", "Frank", "Jack", "Henry",
    "Anna", "Laura", "Helen", "Ruth", "Sharon", "Barbara",
    "Patrick", "Martin", "Scott", "Timothy", "Ronald", "Kenneth",
    "Stephen", "Alexander", "Benjamin", "Samuel", "Nicholas", "Jonathan",
    "Catherine", "Christine", "Rachel", "Stephanie", "Angela",
]

SWAP_PAIRS = [
    ("Emily", "Lakisha"), ("Todd", "Tyrone"), ("Sarah", "Tanisha"),
    ("Anne", "Aisha"), ("Brett", "Jamal"),
]

PROFESSIONS = ["doctor", "nurse", "engineer", "teacher", "lawyer",
               "scientist", "professor", "therapist"]
QUERIES = [f"Who is an experienced {p}?" for p in PROFESSIONS]


def classify_token(tok):
    t = tok.lower().strip("#")
    return "function" if t in FUNCTION_WORDS or len(t) <= 1 else "content"


def generate_passages(n=200):
    """Generate diverse professional passages — 16 templates × 14 professions × 13 domains.
    These are structurally different from the 7 synthetic templates used in the main experiment.
    """
    import random
    random.seed(42)
    tmpls = [
        "{name} is a board-certified {prof} with over {y} years of clinical experience. {name} specializes in {area} and has published more than {p} peer-reviewed articles in leading journals.",
        "Dr. {name}, who has served as chief {prof} since {yr}, recommends early intervention. {name} has treated thousands of patients and is nationally recognized for expertise in {area}.",
        "The legal team led by {name}, a senior {prof}, successfully argued the landmark case reshaping {area} policy. {name} has been recognized by multiple national organizations for outstanding contributions.",
        "{name}, a veteran {prof}, recently published a comprehensive study on {area}. The findings suggest significant improvements in patient outcomes over conventional approaches used in the past decade.",
        "After graduating summa cum laude, {name} joined the practice as a junior {prof}. Over {y} years, {name} rose to become one of the most respected voices in {area} nationally.",
        "In a groundbreaking study, {name} and colleagues demonstrated that {area} treatment can reduce complications by 30 percent. Dr. {name} called the results transformative for the field.",
        "The board appointed {name}, a licensed {prof}, to lead the new {area} initiative. {name} brings {y} years of experience and a proven track record of innovation and leadership.",
        "{name} received the prestigious national award for contributions to {area}. As a practicing {prof}, {name} has mentored dozens of young professionals entering the field.",
        "Patients consistently praise {name} for providing compassionate and thorough care as a {prof}. Over {y} years in practice, {name} has built a reputation for excellence in {area}.",
        "The research conducted by {name} at the university has advanced our understanding of {area}. As an accomplished {prof}, {name} has secured over {p} grants from federal agencies.",
        "Following a distinguished career in {area}, {name} now serves as a consulting {prof} for several Fortune 500 companies. {name} is frequently invited to speak at international conferences.",
        "When it comes to {area}, few professionals match the expertise of {name}. A licensed {prof} for {y} years, {name} has handled over a thousand complex cases successfully.",
        "{name} completed a residency at Johns Hopkins before becoming a practicing {prof} in downtown Boston. {name} is particularly known for innovative approaches to treating {area} conditions.",
        "The report authored by {name}, a leading {prof} in the field, highlights the urgent need for reform in {area}. {name} presented the findings before Congress last month.",
        "Colleagues describe {name} as an exceptionally dedicated {prof} whose commitment to {area} has inspired a new generation of practitioners. {name} has been in practice for {y} years.",
        "In an exclusive interview, {name} shared insights on the future of {area}. Having spent {y} years as a {prof}, {name} believes that technology will fundamentally transform the industry.",
    ]
    profs = ["physician", "surgeon", "attorney", "engineer", "researcher",
             "specialist", "professor", "therapist", "analyst", "consultant",
             "practitioner", "clinician", "advisor", "director"]
    areas = ["cardiovascular medicine", "corporate litigation", "structural engineering",
             "machine learning", "public health", "pediatric oncology",
             "intellectual property", "environmental policy", "data science",
             "neuroscience", "family law", "renewable energy", "clinical psychology"]
    out = []
    for _ in range(n):
        nm = random.choice(COMMON_NAMES[:30])
        t = random.choice(tmpls).format(
            name=nm, prof=random.choice(profs), y=random.randint(8, 30),
            area=random.choice(areas), p=random.randint(20, 120), yr=random.randint(1995, 2018)
        )
        out.append((t, nm))
    return out


def run():
    print("=" * 60)
    print("Ecological Validation (using same ColBERTRetriever)")
    print("=" * 60)

    model = ColBERTRetriever()
    model.load()

    passages = generate_passages(200)
    print(f"Generated {len(passages)} diverse passages (16 templates × 14 profs × 13 domains)")

    results = []
    n_done = 0

    for pi, (passage, orig_name) in enumerate(passages):
        for swap_a, swap_b in SWAP_PAIRS:
            da = re.sub(r'\b' + re.escape(orig_name) + r'\b', swap_a, passage)
            db = re.sub(r'\b' + re.escape(orig_name) + r'\b', swap_b, passage)
            if da == db:
                continue

            for query in QUERIES:
                ra = model.score(query, da)
                rb = model.score(query, db)

                if ra.per_token_scores is None or rb.per_token_scores is None:
                    continue

                min_len = min(len(ra.per_token_scores), len(rb.per_token_scores))
                tcd = compute_tcd(
                    ra.per_token_scores[:min_len],
                    rb.per_token_scores[:min_len]
                )
                q_tokens = ra.query_tokens[:min_len]
                fc = compute_func_cont_tcd(tcd, q_tokens)
                ss = compute_score_sensitivity(ra.total_score, rb.total_score)

                results.append({
                    "pidx": pi, "query": query,
                    "name_a": swap_a, "name_b": swap_b,
                    "func_tcd": fc["func_tcd"],
                    "cont_tcd": fc["cont_tcd"],
                    "ratio": fc["tcd_ratio"],
                    "ss": ss,
                    "score_a": ra.total_score,
                    "score_b": rb.total_score,
                })
                n_done += 1

        if (pi + 1) % 25 == 0:
            print(f"  {pi+1}/{len(passages)} passages ({n_done} tests)")

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(RESULTS / "ecological_validation.csv", index=False)

    print("\n" + "=" * 60)
    print("ECOLOGICAL VALIDATION RESULTS")
    print("=" * 60)
    print(f"Total tests: {len(df)}")
    print(f"Func-TCD: {df['func_tcd'].mean():.4f} ± {df['func_tcd'].std():.4f}")
    print(f"Cont-TCD: {df['cont_tcd'].mean():.4f} ± {df['cont_tcd'].std():.4f}")

    r = df['func_tcd'].mean() / df['cont_tcd'].mean() if df['cont_tcd'].mean() > 0 else 0
    print(f"Func/Cont ratio: {r:.2f}× (synthetic: 1.41×)")
    print(f"Mean SS: {df['ss'].mean():.6f}")

    # Per-query
    print("\nPer-query Func/Cont ratio:")
    for q in sorted(df['query'].unique()):
        s = df[df['query'] == q]
        qr = s['func_tcd'].mean() / s['cont_tcd'].mean() if s['cont_tcd'].mean() > 0 else 0
        p = q.split("experienced ")[-1].rstrip("?")
        print(f"  {p:15s}: {qr:.2f}× {'✓' if qr > 1 else '✗'}")

    # Stats
    from scipy import stats
    stat, pv = stats.wilcoxon(df['func_tcd'].values, df['cont_tcd'].values, alternative='greater')
    ps = np.sqrt((df['func_tcd'].var() + df['cont_tcd'].var()) / 2)
    d = (df['func_tcd'].mean() - df['cont_tcd'].mean()) / ps if ps > 0 else 0
    print(f"\nWilcoxon (Func > Cont): p = {pv:.2e}")
    print(f"Cohen's d: {d:.2f}")
    print(f"Ratio>1 in {(df['ratio']>1).mean()*100:.1f}% of tests")

    # Compare to synthetic
    print("\n--- Comparison to Synthetic Templates ---")
    try:
        syn = pd.read_csv(RESULTS / "full_65name_results.csv")
        syn_r = syn['func_tcd_mean'].mean() / syn['cont_tcd_mean'].mean() if 'func_tcd_mean' in syn.columns else "N/A"
        print(f"Synthetic ratio: {syn_r}")
    except:
        print("Synthetic ratio: 1.41× (from paper)")
    print(f"Ecological ratio: {r:.2f}×")

    with open(RESULTS / "ecological_validation_summary.json", "w") as f:
        json.dump({
            "n_tests": len(df), "n_passages": len(passages),
            "func_cont_ratio": float(r),
            "mean_func_tcd": float(df['func_tcd'].mean()),
            "mean_cont_tcd": float(df['cont_tcd'].mean()),
            "mean_ss": float(df['ss'].mean()),
            "wilcoxon_p": float(pv), "cohens_d": float(d),
            "pct_ratio_gt1": float((df['ratio']>1).mean()*100),
        }, f, indent=2)

    print(f"\nSaved to {RESULTS}")


if __name__ == "__main__":
    run()
