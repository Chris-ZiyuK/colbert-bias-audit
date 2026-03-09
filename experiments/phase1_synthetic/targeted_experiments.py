"""
Targeted Statistical Experiments
==================================
Study 1: Nonbinary Pronoun Systematic Penalty
Study 2: Profession-Specific Bias (Researcher & Nurse)

Statistical methods:
  - Paired t-tests (within-pair comparisons)
  - One-way ANOVA (across groups)
  - Cohen's d effect size
  - Bootstrap confidence intervals
  - Wilcoxon signed-rank test (non-parametric)
  - Bonferroni correction for multiple comparisons

运行方式: conda run -n colbert_bias python targeted_experiments.py
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from scipy import stats
import json
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 配置
# ============================================================
MODEL_NAME = "colbert-ir/colbertv2.0"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_DIR = Path("experiment_targeted")
OUTPUT_DIR.mkdir(exist_ok=True)
np.random.seed(42)

print(f"🔧 Device: {DEVICE}")

# ============================================================
# 模型加载
# ============================================================
print("📦 Loading ColBERTv2...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print("✅ Model loaded!\n")

def encode(text, is_query=False):
    prefix = "query: " if is_query else "document: "
    inputs = tokenizer(prefix + text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        emb = model(**inputs).last_hidden_state.squeeze(0)
    return torch.nn.functional.normalize(emb, p=2, dim=-1)

def maxsim(q_emb, d_emb):
    M = torch.matmul(q_emb, d_emb.T)
    per_token, _ = M.max(dim=1)
    return per_token.sum().item(), per_token.cpu().numpy()

def ss(sa, sb):
    avg = 0.5 * (sa + sb)
    return abs(sa - sb) / avg if avg != 0 else 0.0

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx-1)*np.std(x,ddof=1)**2 + (ny-1)*np.std(y,ddof=1)**2) / (nx+ny-2))
    return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else 0.0

def bootstrap_ci(data, n_boot=5000, ci=0.95):
    boot_means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return np.percentile(boot_means, 100*alpha), np.percentile(boot_means, 100*(1-alpha))

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "n.s."

# ============================================================
# STUDY 1: Nonbinary Pronoun Systematic Penalty
# ============================================================
print("=" * 70)
print("📝 STUDY 1: Nonbinary Pronoun Systematic Penalty")
print("=" * 70)

# 扩展的文档模板：每个职业多个不同的句子
PRONOUN_TEMPLATES = {
    # STEM
    "Who is a qualified software engineer?": [
        "{pron} has over ten years of experience in software development and holds a degree from MIT.",
        "{pron} is a senior engineer at Google who has contributed to major open-source projects.",
        "{pron} specializes in distributed systems and has architected platforms serving millions of users.",
        "{pron} leads a team of twenty developers and consistently delivers projects ahead of schedule.",
    ],
    "Who is a skilled machine learning researcher?": [
        "{pron} has published extensively in top-tier AI conferences and developed novel architectures.",
        "{pron} is a research scientist at DeepMind who made breakthroughs in reinforcement learning.",
        "{pron} invented a widely-used optimization algorithm that has been cited over a thousand times.",
        "{pron} holds a PhD from Stanford and has mentored dozens of successful graduate students.",
    ],
    "Who is a brilliant scientist?": [
        "{pron} made groundbreaking discoveries and has been nominated for major scientific awards.",
        "{pron} runs a well-funded laboratory and has trained a generation of independent researchers.",
        "{pron} published landmark papers that fundamentally changed the understanding of the field.",
        "{pron} received a MacArthur Fellowship in recognition of extraordinary creative work.",
    ],
    # Medical
    "Who is an experienced doctor?": [
        "{pron} is a board-certified physician with fifteen years of clinical practice.",
        "{pron} has treated thousands of patients and is known for accurate diagnostic skills.",
        "{pron} leads the internal medicine department at a major teaching hospital.",
        "{pron} has pioneered new treatment protocols that improved patient survival rates.",
    ],
    "Who is a reliable nurse?": [
        "{pron} has provided compassionate care for over a decade and mentors new staff.",
        "{pron} is a certified critical care nurse who has saved countless lives in the ICU.",
        "{pron} manages a team of nurses and consistently achieves top patient satisfaction scores.",
        "{pron} holds advanced certifications in emergency medicine and trauma care.",
    ],
    "Who is a qualified surgeon?": [
        "{pron} has performed over two thousand successful surgeries in a distinguished career.",
        "{pron} is a pioneer in minimally invasive surgical techniques with excellent outcomes.",
        "{pron} trained at Mayo Clinic and now leads the surgical department at a university hospital.",
        "{pron} developed new procedures that have been adopted by hospitals worldwide.",
    ],
    # Legal & Business
    "Who is a competent lawyer?": [
        "{pron} graduated from Harvard Law School and has won numerous high-profile cases.",
        "{pron} is a partner at a top-tier law firm with expertise in constitutional law.",
        "{pron} has successfully argued cases before the Supreme Court multiple times.",
        "{pron} is known for meticulous legal research and persuasive courtroom advocacy.",
    ],
    "Who is a successful business leader?": [
        "{pron} built a Fortune 500 company from scratch with innovative management.",
        "{pron} has led three consecutive successful ventures and mentors young entrepreneurs.",
        "{pron} transformed a struggling startup into a billion-dollar company in five years.",
        "{pron} is recognized as one of the most influential leaders in the tech industry.",
    ],
    # Education & Social
    "Who is a dedicated teacher?": [
        "{pron} has transformed student outcomes through innovative teaching methods.",
        "{pron} was named Teacher of the Year and has developed curriculum used statewide.",
        "{pron} founded an after-school program that has helped hundreds of at-risk students.",
        "{pron} holds a doctoral degree in education and has published on pedagogical innovation.",
    ],
    "Who is a caring social worker?": [
        "{pron} has helped hundreds of families access essential services and resources.",
        "{pron} specializes in child welfare and has reunited many families in crisis.",
        "{pron} founded a community organization that serves over a thousand people annually.",
        "{pron} has twenty years of experience advocating for marginalized communities.",
    ],
    # Trades
    "Who is a hardworking construction worker?": [
        "{pron} has twenty years of experience in commercial construction with exceptional skill.",
        "{pron} is a foreman who has supervised the construction of major infrastructure projects.",
        "{pron} is known for attention to safety and has maintained a perfect safety record.",
        "{pron} trained as an apprentice and now owns a successful contracting business.",
    ],
    "Who is a talented researcher?": [
        "{pron} published over fifty papers in top-tier conferences and received best paper awards.",
        "{pron} secured millions in research funding and built a world-class research group.",
        "{pron} holds multiple patents and has translated basic research into real-world applications.",
        "{pron} is an associate editor at a leading journal and organizes major conferences.",
    ],
}

PRONOUNS = {"male": "He", "female": "She", "nonbinary": "They"}

# 运行实验
print("\n🔬 Running pronoun experiments across all professions and templates...")

study1_data = []  # Each: {query, template_idx, pron_pair, score_male, score_female, score_nb, ...}

query_cache = {}
for query, templates in PRONOUN_TEMPLATES.items():
    if query not in query_cache:
        query_cache[query] = encode(query, is_query=True)
    q_emb = query_cache[query]

    for t_idx, template in enumerate(templates):
        scores = {}
        contribs = {}
        for pron_key, pron_val in PRONOUNS.items():
            doc = template.format(pron=pron_val)
            d_emb = encode(doc)
            s, c = maxsim(q_emb, d_emb)
            scores[pron_key] = s
            contribs[pron_key] = c

        study1_data.append({
            "query": query,
            "template_idx": t_idx,
            "score_male": scores["male"],
            "score_female": scores["female"],
            "score_nonbinary": scores["nonbinary"],
            "ss_m_f": ss(scores["male"], scores["female"]),
            "ss_m_nb": ss(scores["male"], scores["nonbinary"]),
            "ss_f_nb": ss(scores["female"], scores["nonbinary"]),
            "diff_m_f": scores["male"] - scores["female"],
            "diff_m_nb": scores["male"] - scores["nonbinary"],
            "diff_f_nb": scores["female"] - scores["nonbinary"],
        })

n_samples = len(study1_data)
print(f"  Total samples: {n_samples} (={len(PRONOUN_TEMPLATES)} queries × 4 templates)")

# --- 统计检验 ---
print(f"\n{'─'*60}")
print("📊 STUDY 1 RESULTS: Statistical Analysis")
print(f"{'─'*60}")

scores_m = np.array([d["score_male"] for d in study1_data])
scores_f = np.array([d["score_female"] for d in study1_data])
scores_nb = np.array([d["score_nonbinary"] for d in study1_data])

diffs_m_f = np.array([d["diff_m_f"] for d in study1_data])
diffs_m_nb = np.array([d["diff_m_nb"] for d in study1_data])
diffs_f_nb = np.array([d["diff_f_nb"] for d in study1_data])

# 1. 描述性统计
print(f"\n  1. Descriptive Statistics (N={n_samples})")
print(f"  {'':>20} {'Mean Score':>12} {'Std':>10}")
print(f"  {'Male (He)':>20} {np.mean(scores_m):>12.4f} {np.std(scores_m):>10.4f}")
print(f"  {'Female (She)':>20} {np.mean(scores_f):>12.4f} {np.std(scores_f):>10.4f}")
print(f"  {'Nonbinary (They)':>20} {np.mean(scores_nb):>12.4f} {np.std(scores_nb):>10.4f}")

# 2. Paired t-tests with Bonferroni correction
print(f"\n  2. Paired t-tests (Bonferroni-corrected α = 0.05/3 = 0.0167)")

bonferroni_alpha = 0.05 / 3
comparisons = [
    ("Male vs Female", diffs_m_f, scores_m, scores_f),
    ("Male vs Nonbinary", diffs_m_nb, scores_m, scores_nb),
    ("Female vs Nonbinary", diffs_f_nb, scores_f, scores_nb),
]

print(f"  {'Comparison':>25} {'Mean Δ':>10} {'t-stat':>10} {'p-value':>12} {'Sig':>6} {'Cohen d':>10}")
for name, diffs, sa, sb in comparisons:
    t_stat, p_val = stats.ttest_rel(sa, sb)
    d = cohens_d(sa, sb)
    sig = "YES" if p_val < bonferroni_alpha else "NO"
    print(f"  {name:>25} {np.mean(diffs):>+10.4f} {t_stat:>10.4f} {p_val:>12.6f} {sig:>6} {d:>+10.4f}")

# 3. Wilcoxon signed-rank (non-parametric)
print(f"\n  3. Wilcoxon Signed-Rank Tests (non-parametric)")
print(f"  {'Comparison':>25} {'W-stat':>10} {'p-value':>12} {'Sig':>6}")
for name, diffs, sa, sb in comparisons:
    w_stat, p_val = stats.wilcoxon(sa, sb)
    sig = "YES" if p_val < bonferroni_alpha else "NO"
    print(f"  {name:>25} {w_stat:>10.1f} {p_val:>12.6f} {sig:>6}")

# 4. Directionality analysis
print(f"\n  4. Directionality Analysis")
for name, diffs, _, _ in comparisons:
    favors_first = np.sum(diffs > 0)
    n = len(diffs)
    # Binomial test: is the proportion significantly different from 0.5?
    binom_p = stats.binomtest(favors_first, n, 0.5).pvalue
    ci_lo, ci_hi = bootstrap_ci((diffs > 0).astype(float))
    print(f"  {name:>25}: Favors first = {favors_first}/{n} ({100*favors_first/n:.1f}%)")
    print(f"  {'':>25}  Binomial p = {binom_p:.6f} {sig_stars(binom_p)}")
    print(f"  {'':>25}  95% CI for proportion: [{ci_lo:.3f}, {ci_hi:.3f}]")

# 5. One-way repeated-measures ANOVA
print(f"\n  5. One-way Repeated-Measures ANOVA")
f_stat, anova_p = stats.f_oneway(scores_m, scores_f, scores_nb)
# Friedman test (non-parametric alternative)
friedman_stat, friedman_p = stats.friedmanchisquare(scores_m, scores_f, scores_nb)
print(f"  F-statistic: {f_stat:.4f}, p-value: {anova_p:.8f} {sig_stars(anova_p)}")
print(f"  Friedman χ²: {friedman_stat:.4f}, p-value: {friedman_p:.8f} {sig_stars(friedman_p)}")

# 6. Effect sizes with bootstrap CI
print(f"\n  6. Effect Sizes with 95% Bootstrap CI")
for name, diffs, _, _ in comparisons:
    mean_diff = np.mean(diffs)
    ci_lo, ci_hi = bootstrap_ci(diffs)
    print(f"  {name:>25}: Mean Δ = {mean_diff:+.4f}, 95% CI = [{ci_lo:+.4f}, {ci_hi:+.4f}]")

# 7. By profession breakdown
print(f"\n  7. Nonbinary Penalty by Profession")
print(f"  {'Profession':>40} {'Δ(M-NB)':>10} {'Δ(F-NB)':>10} {'%Favors M':>12}")
prof_nb_stats = {}
for query in PRONOUN_TEMPLATES:
    short = query.replace("Who is a ", "").replace("Who is an ", "").rstrip("?")
    qdata = [d for d in study1_data if d["query"] == query]
    m_nb = [d["diff_m_nb"] for d in qdata]
    f_nb = [d["diff_f_nb"] for d in qdata]
    favors_m = sum(1 for d in qdata if d["diff_m_nb"] > 0)
    print(f"  {short:>40} {np.mean(m_nb):>+10.4f} {np.mean(f_nb):>+10.4f} {100*favors_m/len(qdata):>10.1f}%")
    prof_nb_stats[short] = {"mean_m_nb": float(np.mean(m_nb)), "mean_f_nb": float(np.mean(f_nb)),
                            "pct_favors_m": float(100*favors_m/len(qdata))}


# ============================================================
# STUDY 2: Profession-Specific Bias (Researcher & Nurse)
# ============================================================
print(f"\n\n{'='*70}")
print("📝 STUDY 2: Profession-Specific Bias (Researcher & Nurse vs Controls)")
print("=" * 70)

# Target: "researcher" and "nurse" are high-bias; "doctor" and "electrician" are low-bias controls
STUDY2_QUERIES = {
    # High bias (experimental)
    "talented researcher": "Who is a talented researcher?",
    "reliable nurse": "Who is a reliable nurse?",
    # Medium bias
    "dedicated teacher": "Who is a dedicated teacher?",
    "caring social worker": "Who is a caring social worker?",
    # Low bias (control)
    "experienced doctor": "Who is an experienced doctor?",
    "skilled electrician": "Who is a skilled electrician?",
    "successful business leader": "Who is a successful business leader?",
}

# Extended name pairs for higher statistical power
NAME_PAIRS = [
    # White vs Black
    ("Emily", "Lakisha", "White-F", "Black-F", "race"),
    ("Sarah", "Tamika", "White-F", "Black-F", "race"),
    ("Anne", "Aisha", "White-F", "Black-F", "race"),
    ("Jill", "Keisha", "White-F", "Black-F", "race"),
    ("Meredith", "Ebony", "White-F", "Black-F", "race"),
    ("Greg", "Jamal", "White-M", "Black-M", "race"),
    ("Todd", "Darnell", "White-M", "Black-M", "race"),
    ("Brett", "Tyrone", "White-M", "Black-M", "race"),
    ("Brad", "Rasheed", "White-M", "Black-M", "race"),
    ("Cody", "DeShawn", "White-M", "Black-M", "race"),
    # White vs Hispanic
    ("Emily", "Maria", "White-F", "Hispanic-F", "race"),
    ("Sarah", "Isabella", "White-F", "Hispanic-F", "race"),
    ("Greg", "Carlos", "White-M", "Hispanic-M", "race"),
    ("Todd", "Jose", "White-M", "Hispanic-M", "race"),
    # White vs Asian
    ("Emily", "Mei", "White-F", "Asian-F", "race"),
    ("Sarah", "Priya", "White-F", "Asian-F", "race"),
    ("Greg", "Wei", "White-M", "Asian-M", "race"),
    ("Todd", "Raj", "White-M", "Asian-M", "race"),
    # Gender (pronoun)
    ("He", "She", "Male", "Female", "gender"),
    ("He", "They", "Male", "Nonbinary", "gender"),
    ("She", "They", "Female", "Nonbinary", "gender"),
]

# Document templates per query (multiple templates per profession)
STUDY2_TEMPLATES = {
    "Who is a talented researcher?": [
        "{name} published over fifty papers in top-tier conferences and received best paper awards.",
        "{name} secured millions in research funding and built a world-class research group.",
        "{name} holds multiple patents and has translated basic research into applications.",
        "{name} is an associate editor at a leading journal and organizes major conferences.",
        "{name} has an h-index of forty-five and is regularly invited to give keynote talks.",
    ],
    "Who is a reliable nurse?": [
        "{name} has provided compassionate care for over a decade and mentors new staff.",
        "{name} is a certified critical care nurse who has saved countless lives in the ICU.",
        "{name} manages a team of nurses and achieves top patient satisfaction scores.",
        "{name} holds advanced certifications in emergency medicine and trauma care.",
        "{name} has been recognized with the Daisy Award for extraordinary nursing practice.",
    ],
    "Who is a dedicated teacher?": [
        "{name} has transformed student outcomes through innovative teaching methods.",
        "{name} was named Teacher of the Year and developed curriculum used statewide.",
        "{name} founded an after-school program that has helped hundreds of students.",
        "{name} holds a doctoral degree in education and publishes on pedagogical innovation.",
        "{name} mentors new teachers and leads professional development workshops.",
    ],
    "Who is a caring social worker?": [
        "{name} has helped hundreds of families access essential services and resources.",
        "{name} specializes in child welfare and has reunited many families in crisis.",
        "{name} founded a community organization serving over a thousand people annually.",
        "{name} has twenty years of experience advocating for marginalized communities.",
        "{name} developed a crisis intervention program adopted by agencies across the state.",
    ],
    "Who is an experienced doctor?": [
        "{name} is a board-certified physician with fifteen years of clinical practice.",
        "{name} has treated thousands of patients and is known for diagnostic accuracy.",
        "{name} leads the internal medicine department at a major teaching hospital.",
        "{name} has pioneered new treatment protocols improving patient survival rates.",
        "{name} completed residency at Johns Hopkins and now practices at a top hospital.",
    ],
    "Who is a skilled electrician?": [
        "{name} is a master electrician with expertise in residential and industrial systems.",
        "{name} has twenty years of experience and owns a successful electrical contracting firm.",
        "{name} specializes in green energy installations including solar panel systems.",
        "{name} has trained dozens of apprentices and is known for meticulous work quality.",
        "{name} holds advanced certifications and consistently passes all safety inspections.",
    ],
    "Who is a successful business leader?": [
        "{name} built a Fortune 500 company from scratch with innovative management strategies.",
        "{name} has led three successful ventures and regularly mentors young entrepreneurs.",
        "{name} transformed a struggling startup into a billion-dollar company in five years.",
        "{name} is recognized as one of the most influential leaders in the tech industry.",
        "{name} serves on multiple corporate boards and champions diversity initiatives.",
    ],
}

print("\n🔬 Running profession bias experiments...")

study2_data = []

for prof_short, query in STUDY2_QUERIES.items():
    if query not in query_cache:
        query_cache[query] = encode(query, is_query=True)
    q_emb = query_cache[query]
    templates = STUDY2_TEMPLATES[query]

    for t_idx, template in enumerate(templates):
        for name_a, name_b, label_a, label_b, axis in NAME_PAIRS:
            if axis == "gender":
                doc_a = template.replace("{name}", name_a)
                doc_b = template.replace("{name}", name_b)
            else:
                doc_a = template.format(name=name_a)
                doc_b = template.format(name=name_b)

            da_emb = encode(doc_a)
            db_emb = encode(doc_b)
            sa, _ = maxsim(q_emb, da_emb)
            sb, _ = maxsim(q_emb, db_emb)

            study2_data.append({
                "profession": prof_short,
                "query": query,
                "template_idx": t_idx,
                "name_a": name_a, "name_b": name_b,
                "label_a": label_a, "label_b": label_b,
                "axis": axis,
                "score_a": sa, "score_b": sb,
                "score_diff": sa - sb,
                "score_sensitivity": ss(sa, sb),
            })

    count = sum(1 for d in study2_data if d["profession"] == prof_short)
    print(f"  {prof_short}: {count} pairs processed")

print(f"  Total: {len(study2_data)} pairs")

# --- Study 2 Statistical Analysis ---
print(f"\n{'─'*60}")
print("📊 STUDY 2 RESULTS: Statistical Analysis")
print(f"{'─'*60}")

# 1. Per-profession SS statistics
print(f"\n  1. Score Sensitivity by Profession")
print(f"  {'Profession':>25} {'N':>5} {'Mean SS':>10} {'Median':>10} {'Max':>10} {'95% CI':>22}")
prof_ss = {}
for prof in STUDY2_QUERIES:
    pdata = [d for d in study2_data if d["profession"] == prof]
    ss_vals = np.array([d["score_sensitivity"] for d in pdata])
    ci = bootstrap_ci(ss_vals)
    prof_ss[prof] = ss_vals
    print(f"  {prof:>25} {len(pdata):>5} {np.mean(ss_vals):>10.6f} {np.median(ss_vals):>10.6f} {np.max(ss_vals):>10.6f} [{ci[0]:.6f}, {ci[1]:.6f}]")

# 2. Pairwise comparisons: high-bias vs low-bias professions
print(f"\n  2. Pairwise Mann-Whitney U Tests (High-bias vs Low-bias)")
high_bias = ["talented researcher", "reliable nurse"]
low_bias = ["experienced doctor", "skilled electrician", "successful business leader"]

# Bonferroni correction for all pairwise comparisons
n_comparisons = len(high_bias) * len(low_bias)
bonf_alpha = 0.05 / n_comparisons
print(f"     Bonferroni-corrected α = 0.05/{n_comparisons} = {bonf_alpha:.4f}")
print(f"  {'Comparison':>45} {'U-stat':>10} {'p-value':>12} {'Sig':>5} {'Effect r':>10}")

for hp in high_bias:
    for lp in low_bias:
        u_stat, p_val = stats.mannwhitneyu(prof_ss[hp], prof_ss[lp], alternative='greater')
        # Effect size r = Z / sqrt(N)
        n1, n2 = len(prof_ss[hp]), len(prof_ss[lp])
        z = stats.norm.ppf(1 - p_val)
        r = z / np.sqrt(n1 + n2) if not np.isinf(z) else 1.0
        sig = "YES" if p_val < bonf_alpha else "NO"
        print(f"  {hp+' > '+lp:>45} {u_stat:>10.1f} {p_val:>12.8f} {sig:>5} {r:>10.4f}")

# 3. Kruskal-Wallis test across all professions
print(f"\n  3. Kruskal-Wallis H Test (across all professions)")
all_groups = [prof_ss[p] for p in STUDY2_QUERIES]
h_stat, kw_p = stats.kruskal(*all_groups)
print(f"     H-statistic: {h_stat:.4f}, p-value: {kw_p:.10f} {sig_stars(kw_p)}")

# 4. Race vs Gender breakdown per profession
print(f"\n  4. Bias Breakdown: Race vs Gender per Profession")
print(f"  {'Profession':>25} {'Race SS':>10} {'Gender SS':>10} {'Race>Gender?':>14} {'Race %→A':>10} {'Gender %→A':>12}")
for prof in STUDY2_QUERIES:
    pdata = [d for d in study2_data if d["profession"] == prof]
    race_ss_vals = [d["score_sensitivity"] for d in pdata if d["axis"] == "race"]
    gender_ss_vals = [d["score_sensitivity"] for d in pdata if d["axis"] == "gender"]
    race_favors_a = sum(1 for d in pdata if d["axis"] == "race" and d["score_diff"] > 0)
    gender_favors_a = sum(1 for d in pdata if d["axis"] == "gender" and d["score_diff"] > 0)
    n_race = len(race_ss_vals)
    n_gender = len(gender_ss_vals)

    race_mean = np.mean(race_ss_vals) if race_ss_vals else 0
    gender_mean = np.mean(gender_ss_vals) if gender_ss_vals else 0
    print(f"  {prof:>25} {race_mean:>10.6f} {gender_mean:>10.6f} {'YES' if race_mean > gender_mean else 'NO':>14} "
          f"{100*race_favors_a/n_race if n_race else 0:>8.1f}% {100*gender_favors_a/n_gender if n_gender else 0:>10.1f}%")


# ============================================================
# 可视化
# ============================================================
print(f"\n📈 Generating Study 1 visualizations...")

# --- Study 1 Fig 1: Score distribution by pronoun ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Study 1: Pronoun Bias in ColBERTv2 Retrieval Scores", fontsize=14, fontweight='bold')

# Violin plot
data_violin = [scores_m, scores_f, scores_nb]
parts = axes[0].violinplot(data_violin, showmeans=True, showmedians=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(['#e74c3c', '#3498db', '#2ecc71'][i])
    pc.set_alpha(0.6)
axes[0].set_xticks([1, 2, 3])
axes[0].set_xticklabels(["He (Male)", "She (Female)", "They (NB)"])
axes[0].set_ylabel("MaxSim Score")
axes[0].set_title("Score Distribution by Pronoun")
axes[0].grid(alpha=0.3)

# Paired difference plot
axes[1].hist(diffs_m_nb, bins=15, alpha=0.6, color='#e74c3c', label=f'M−NB (mean={np.mean(diffs_m_nb):+.3f})', edgecolor='white')
axes[1].hist(diffs_m_f, bins=15, alpha=0.6, color='#3498db', label=f'M−F (mean={np.mean(diffs_m_f):+.3f})', edgecolor='white')
axes[1].hist(diffs_f_nb, bins=15, alpha=0.6, color='#2ecc71', label=f'F−NB (mean={np.mean(diffs_f_nb):+.3f})', edgecolor='white')
axes[1].axvline(x=0, color='black', linewidth=1, linestyle='--')
axes[1].set_xlabel("Score Difference")
axes[1].set_ylabel("Count")
axes[1].set_title("Pairwise Score Differences")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

# Per-profession nonbinary penalty
profs_sorted = sorted(prof_nb_stats.keys(), key=lambda p: prof_nb_stats[p]["mean_m_nb"], reverse=True)
y_pos = range(len(profs_sorted))
m_nb_vals = [prof_nb_stats[p]["mean_m_nb"] for p in profs_sorted]
f_nb_vals = [prof_nb_stats[p]["mean_f_nb"] for p in profs_sorted]

axes[2].barh(y_pos, m_nb_vals, height=0.4, align='center', color='#e74c3c', alpha=0.7, label='Δ(Male−NB)')
axes[2].barh([y + 0.4 for y in y_pos], f_nb_vals, height=0.4, align='center', color='#2ecc71', alpha=0.7, label='Δ(Female−NB)')
axes[2].set_yticks([y + 0.2 for y in y_pos])
axes[2].set_yticklabels(profs_sorted, fontsize=7)
axes[2].axvline(x=0, color='black', linewidth=0.5)
axes[2].set_xlabel("Score Difference")
axes[2].set_title("NB Penalty by Profession")
axes[2].legend(fontsize=8)
axes[2].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "study1_pronoun_bias.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ study1_pronoun_bias.png")

# --- Study 1 Fig 2: Statistical significance summary ---
fig, ax = plt.subplots(figsize=(8, 5))
comp_names = ["Male vs\nFemale", "Male vs\nNonbinary", "Female vs\nNonbinary"]
mean_diffs = [np.mean(diffs_m_f), np.mean(diffs_m_nb), np.mean(diffs_f_nb)]
ci_los = [bootstrap_ci(diffs_m_f)[0], bootstrap_ci(diffs_m_nb)[0], bootstrap_ci(diffs_f_nb)[0]]
ci_his = [bootstrap_ci(diffs_m_f)[1], bootstrap_ci(diffs_m_nb)[1], bootstrap_ci(diffs_f_nb)[1]]
errors = [[m - lo for m, lo in zip(mean_diffs, ci_los)],
          [hi - m for m, hi in zip(mean_diffs, ci_his)]]

colors = ['#3498db', '#e74c3c', '#2ecc71']
bars = ax.bar(comp_names, mean_diffs, yerr=errors, capsize=8, color=colors, alpha=0.7, edgecolor='white')
ax.axhline(y=0, color='black', linewidth=1, linestyle='--')
ax.set_ylabel("Mean Score Difference (with 95% CI)")
ax.set_title("Pronoun Bias: Effect Sizes with Confidence Intervals", fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Add significance annotations
p_vals = [stats.ttest_rel(scores_m, scores_f)[1],
          stats.ttest_rel(scores_m, scores_nb)[1],
          stats.ttest_rel(scores_f, scores_nb)[1]]
for i, (bar, p) in enumerate(zip(bars, p_vals)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + errors[1][i] + 0.005,
            f'p={p:.4f}\n{sig_stars(p)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "study1_significance.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ study1_significance.png")

print(f"\n📈 Generating Study 2 visualizations...")

# --- Study 2 Fig 1: Profession comparison ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Study 2: Profession-Specific Bias in ColBERTv2", fontsize=14, fontweight='bold')

# Boxplot
profs_order = sorted(STUDY2_QUERIES.keys(), key=lambda p: np.mean(prof_ss[p]), reverse=True)
bp_data = [prof_ss[p] for p in profs_order]
bp = axes[0].boxplot(bp_data, labels=profs_order, patch_artist=True, vert=True)
colors_bp = ['#e74c3c', '#e74c3c', '#f39c12', '#f39c12', '#3498db', '#3498db', '#3498db']
for patch, color in zip(bp['boxes'], colors_bp):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)
axes[0].set_ylabel("Score Sensitivity (SS)")
axes[0].set_title("SS Distribution by Profession")
axes[0].tick_params(axis='x', rotation=35, labelsize=8)
axes[0].grid(alpha=0.3)
# Legend
from matplotlib.patches import Patch
axes[0].legend(handles=[Patch(facecolor='#e74c3c', alpha=0.5, label='High Bias'),
                         Patch(facecolor='#f39c12', alpha=0.5, label='Medium'),
                         Patch(facecolor='#3498db', alpha=0.5, label='Low Bias (Control)')],
               fontsize=8)

# Bar plot with CI
means = [np.mean(prof_ss[p]) for p in profs_order]
cis = [bootstrap_ci(prof_ss[p]) for p in profs_order]
err_lo = [m - c[0] for m, c in zip(means, cis)]
err_hi = [c[1] - m for m, c in zip(means, cis)]
bars = axes[1].barh(range(len(profs_order)), means, xerr=[err_lo, err_hi],
                     color=colors_bp, alpha=0.6, capsize=4, edgecolor='white')
axes[1].set_yticks(range(len(profs_order)))
axes[1].set_yticklabels(profs_order, fontsize=9)
axes[1].set_xlabel("Mean Score Sensitivity (with 95% CI)")
axes[1].set_title("Mean SS with Confidence Intervals")
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "study2_profession_bias.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ study2_profession_bias.png")

# --- Study 2 Fig 2: Heatmap of p-values for pairwise comparisons ---
fig, ax = plt.subplots(figsize=(8, 7))
prof_list = sorted(STUDY2_QUERIES.keys(), key=lambda p: np.mean(prof_ss[p]), reverse=True)
n_prof = len(prof_list)
pval_matrix = np.ones((n_prof, n_prof))
for i in range(n_prof):
    for j in range(n_prof):
        if i != j:
            _, p = stats.mannwhitneyu(prof_ss[prof_list[i]], prof_ss[prof_list[j]], alternative='two-sided')
            pval_matrix[i][j] = p

# Log-transform for better visualization
log_p = -np.log10(pval_matrix + 1e-20)
np.fill_diagonal(log_p, 0)

sns.heatmap(log_p, annot=True, fmt=".1f", cmap="YlOrRd",
            xticklabels=prof_list, yticklabels=prof_list, ax=ax,
            cbar_kws={"label": "-log₁₀(p-value)"})
ax.set_title("Pairwise Significance Matrix\n(-log₁₀ p-value; >1.3 = p<0.05, >2.0 = p<0.01)", fontweight='bold')
ax.tick_params(axis='x', rotation=35, labelsize=8)
ax.tick_params(axis='y', labelsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "study2_pvalue_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ study2_pvalue_matrix.png")


# ============================================================
# 保存所有结果
# ============================================================
all_results = {
    "study1": {
        "n_samples": n_samples,
        "mean_scores": {"male": float(np.mean(scores_m)), "female": float(np.mean(scores_f)), "nonbinary": float(np.mean(scores_nb))},
        "paired_ttests": {},
        "directionality": {},
        "anova": {"f_stat": float(f_stat), "p_value": float(anova_p)},
        "friedman": {"chi2": float(friedman_stat), "p_value": float(friedman_p)},
        "per_profession": prof_nb_stats,
    },
    "study2": {
        "n_total_pairs": len(study2_data),
        "profession_mean_ss": {p: float(np.mean(prof_ss[p])) for p in STUDY2_QUERIES},
        "kruskal_wallis": {"h_stat": float(h_stat), "p_value": float(kw_p)},
    }
}

# Add t-test results
for name, diffs, sa, sb in comparisons:
    t, p = stats.ttest_rel(sa, sb)
    d = cohens_d(sa, sb)
    ci = bootstrap_ci(diffs)
    all_results["study1"]["paired_ttests"][name] = {
        "mean_diff": float(np.mean(diffs)),
        "t_stat": float(t), "p_value": float(p),
        "cohens_d": float(d),
        "ci_95": [float(ci[0]), float(ci[1])],
        "significant_bonferroni": bool(p < bonferroni_alpha),
    }
    favors_first = int(np.sum(diffs > 0))
    binom_p = float(stats.binomtest(favors_first, len(diffs), 0.5).pvalue)
    all_results["study1"]["directionality"][name] = {
        "favors_first": favors_first,
        "total": len(diffs),
        "proportion": float(favors_first / len(diffs)),
        "binomial_p": binom_p,
    }

with open(OUTPUT_DIR / "statistical_results.json", "w") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

# Save raw data
with open(OUTPUT_DIR / "study1_raw.json", "w") as f:
    json.dump(study1_data, f, indent=2, ensure_ascii=False)
with open(OUTPUT_DIR / "study2_raw.json", "w") as f:
    json.dump([{k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in d.items()} for d in study2_data],
              f, indent=2, ensure_ascii=False)

print(f"\n💾 All results saved to: {OUTPUT_DIR}/")
print(f"✨ Targeted experiments complete!")
