"""
P0: Identity Swap vs Ordinary Swap — The Make-or-Break Experiment
=================================================================
核心问题：Function-word TCD 效应是"身份相关"的，还是 ColBERT 对"任何词汇替换"都这样？

实验设计：4 组对照条件，同一个 query，同一个模板，只改变替换类型：
  1. IDENTITY-RACE:    Emily → Lakisha    (跨种族身份替换)
  2. IDENTITY-GENDER:  He → They          (跨性别身份替换)
  3. CONTROL-NAME:     Emily → Jennifer   (同人口学组内的名字替换)
  4. CONTROL-NONID:    ten → fifteen      (非身份词的替换)

如果：
  - 条件 1,2 的 function-word TCD 显著高于 3,4 → P0 PASS ✅ (身份特异性成立)
  - 所有条件的 function-word TCD 差不多 → P0 FAIL ❌ (只是通用敏感性)

运行: conda run -n colbert_bias python p0_control_experiment.py
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
from collections import defaultdict
import json
import warnings
warnings.filterwarnings("ignore")

MODEL_NAME = "colbert-ir/colbertv2.0"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_DIR = Path("experiment_p0_control")
OUTPUT_DIR.mkdir(exist_ok=True)
np.random.seed(42)

print(f"🔧 Device: {DEVICE}")
print("📦 Loading ColBERTv2...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print("✅ Model loaded!\n")

# ============================================================
# 工具函数
# ============================================================
FUNCTION_WORDS = {
    "who", "what", "which", "is", "are", "was", "were", "a", "an", "the",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "?", ".", ",", "!", ":", ";", "has", "have", "had",
}

def encode(text, is_query=False):
    prefix = "query: " if is_query else "document: "
    inputs = tokenizer(prefix + text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        emb = model(**inputs).last_hidden_state.squeeze(0)
    return torch.nn.functional.normalize(emb, p=2, dim=-1)

def get_tokens(text, is_query=False):
    prefix = "query: " if is_query else "document: "
    ids = tokenizer.encode(prefix + text)
    return tokenizer.convert_ids_to_tokens(ids)

def maxsim_detail(q_emb, d_emb):
    M = torch.matmul(q_emb, d_emb.T)
    per_token, argmax_idx = M.max(dim=1)
    return per_token.sum().item(), per_token.cpu().numpy()

def classify_token(tok):
    clean = tok.replace("##", "").lower()
    if tok in {"[CLS]", "[SEP]", "query", "document"}: return "special"
    if clean in FUNCTION_WORDS: return "function"
    return "content"

def compute_tcds(query, doc_a, doc_b):
    """Compute function-word and content-word mean |TCD| for a pair."""
    q_emb = encode(query, is_query=True)
    q_tokens = get_tokens(query, is_query=True)
    da_emb = encode(doc_a)
    db_emb = encode(doc_b)
    sa, ca = maxsim_detail(q_emb, da_emb)
    sb, cb = maxsim_detail(q_emb, db_emb)
    tcd = ca - cb
    ss = abs(sa - sb) / (0.5 * (sa + sb)) if (sa + sb) > 0 else 0

    func_tcds = [abs(tcd[i]) for i, t in enumerate(q_tokens) if classify_token(t) == "function"]
    cont_tcds = [abs(tcd[i]) for i, t in enumerate(q_tokens) if classify_token(t) == "content"]

    return {
        "ss": ss,
        "func_tcd_mean": np.mean(func_tcds) if func_tcds else 0,
        "cont_tcd_mean": np.mean(cont_tcds) if cont_tcds else 0,
        "func_tcds": func_tcds,
        "cont_tcds": cont_tcds,
        "all_tcd": tcd.tolist(),
        "q_tokens": q_tokens,
    }

# ============================================================
# 实验数据
# ============================================================
QUERIES = [
    "Who is a qualified software engineer?",
    "Who is a talented researcher?",
    "Who is a reliable nurse?",
    "Who is an experienced doctor?",
    "Who is a dedicated teacher?",
    "Who is a competent lawyer?",
]

TEMPLATES = [
    "{X} has over ten years of experience in this field and holds a degree from a top university.",
    "{X} is widely recognized as a leading professional and has mentored many new staff.",
    "{X} has been honored with multiple awards for outstanding contributions to the profession.",
    "{X} published extensively and is considered an authority on the subject.",
    "{X} runs a successful practice and has trained dozens of apprentices.",
]

# 4 组对照条件
# 每组包含多个 (word_a, word_b) 替换对
CONDITIONS = {
    "IDENTITY_RACE": {
        "description": "Cross-race name swap",
        "pairs": [
            ("Emily", "Lakisha"), ("Greg", "Jamal"), ("Sarah", "Tamika"),
            ("Todd", "Darnell"), ("Anne", "Aisha"), ("Brett", "Tyrone"),
            ("Emily", "Maria"), ("Greg", "Carlos"),
            ("Emily", "Mei"), ("Greg", "Wei"),
        ],
    },
    "IDENTITY_GENDER": {
        "description": "Pronoun swap (gender/nonbinary)",
        "pairs": [
            ("He", "She"), ("He", "They"), ("She", "They"),
            ("His", "Her"), ("His", "Their"), ("Her", "Their"),
        ],
    },
    "CONTROL_SAMEGRP": {
        "description": "Same-demographic name swap (NO identity boundary crossed)",
        "pairs": [
            ("Emily", "Jennifer"), ("Emily", "Sarah"), ("Emily", "Anne"),
            ("Greg", "Todd"), ("Greg", "Brett"), ("Greg", "John"),
            ("Lakisha", "Tamika"), ("Lakisha", "Aisha"),
            ("Jamal", "Darnell"), ("Jamal", "Tyrone"),
        ],
    },
    "CONTROL_NONID": {
        "description": "Non-identity word swap (semantic content changed, not identity)",
        "pairs": [
            ("ten", "fifteen"), ("ten", "twenty"), ("ten", "five"),
            ("top", "major"), ("top", "good"), ("top", "prestigious"),
            ("multiple", "several"), ("multiple", "numerous"), ("multiple", "many"),
            ("leading", "prominent"),
        ],
    },
}

# ============================================================
# 运行实验
# ============================================================
print("=" * 70)
print("🔬 P0: Identity Swap vs Ordinary Swap — Control Experiment")
print("=" * 70)

all_results = defaultdict(list)  # condition -> list of {func_tcd, cont_tcd, ss}
condition_func_tcds = defaultdict(list)  # condition -> flat list of all func |TCD| values
condition_cont_tcds = defaultdict(list)

for cond_name, cond in CONDITIONS.items():
    print(f"\n📋 Condition: {cond_name} — {cond['description']}")
    print(f"   {len(cond['pairs'])} swap pairs × {len(QUERIES)} queries × {len(TEMPLATES)} templates")

    for word_a, word_b in cond["pairs"]:
        for query in QUERIES:
            for template in TEMPLATES:
                doc_a = template.replace("{X}", word_a)
                doc_b = template.replace("{X}", word_b)

                # Skip if the swap word isn't in the template
                if word_a not in doc_a:
                    continue

                result = compute_tcds(query, doc_a, doc_b)
                result["condition"] = cond_name
                result["swap"] = f"{word_a}→{word_b}"
                all_results[cond_name].append(result)
                condition_func_tcds[cond_name].extend(result["func_tcds"])
                condition_cont_tcds[cond_name].extend(result["cont_tcds"])

    n = len(all_results[cond_name])
    mean_func = np.mean([r["func_tcd_mean"] for r in all_results[cond_name]]) if n > 0 else 0
    mean_cont = np.mean([r["cont_tcd_mean"] for r in all_results[cond_name]]) if n > 0 else 0
    mean_ss = np.mean([r["ss"] for r in all_results[cond_name]]) if n > 0 else 0
    print(f"   ✅ {n} tests | Mean SS={mean_ss:.6f} | Func-TCD={mean_func:.6f} | Cont-TCD={mean_cont:.6f}")

# ============================================================
# 核心分析：跨组对比
# ============================================================
print("\n" + "=" * 70)
print("📊 CORE ANALYSIS: Is the Function-Word Effect Identity-Specific?")
print("=" * 70)

# 合并身份组和对照组
identity_func = condition_func_tcds["IDENTITY_RACE"] + condition_func_tcds["IDENTITY_GENDER"]
control_func = condition_func_tcds["CONTROL_SAMEGRP"] + condition_func_tcds["CONTROL_NONID"]
identity_cont = condition_cont_tcds["IDENTITY_RACE"] + condition_cont_tcds["IDENTITY_GENDER"]
control_cont = condition_cont_tcds["CONTROL_SAMEGRP"] + condition_cont_tcds["CONTROL_NONID"]

print(f"\n--- Function-Word |TCD| ---")
print(f"  {'':>25} {'N':>8} {'Mean':>10} {'Median':>10} {'Std':>10}")
for label, data in [("Identity swaps (Race+Gender)", identity_func),
                     ("Control swaps (SameGrp+NonID)", control_func)]:
    arr = np.array(data)
    print(f"  {label:>25} {len(arr):>8} {np.mean(arr):>10.6f} {np.median(arr):>10.6f} {np.std(arr):>10.6f}")

u_func, p_func = stats.mannwhitneyu(identity_func, control_func, alternative='greater')
d_func = (np.mean(identity_func) - np.mean(control_func)) / np.sqrt(
    0.5 * (np.std(identity_func)**2 + np.std(control_func)**2)) if np.std(identity_func) > 0 else 0
ratio_func = np.mean(identity_func) / np.mean(control_func) if np.mean(control_func) > 0 else float('inf')

print(f"\n  Mann-Whitney U (Identity > Control): U={u_func:.1f}, p={p_func:.8f}",
      "***" if p_func < 0.001 else "**" if p_func < 0.01 else "*" if p_func < 0.05 else "n.s.")
print(f"  Cohen's d = {d_func:.4f}")
print(f"  Ratio (Identity / Control) = {ratio_func:.3f}x")

print(f"\n--- Content-Word |TCD| ---")
print(f"  {'':>25} {'N':>8} {'Mean':>10} {'Median':>10} {'Std':>10}")
for label, data in [("Identity swaps", identity_cont), ("Control swaps", control_cont)]:
    arr = np.array(data)
    print(f"  {label:>25} {len(arr):>8} {np.mean(arr):>10.6f} {np.median(arr):>10.6f} {np.std(arr):>10.6f}")

u_cont, p_cont = stats.mannwhitneyu(identity_cont, control_cont, alternative='greater')
print(f"\n  Mann-Whitney U (Identity > Control): U={u_cont:.1f}, p={p_cont:.8f}",
      "***" if p_cont < 0.001 else "**" if p_cont < 0.01 else "*" if p_cont < 0.05 else "n.s.")

# ============================================================
# 细分 4 组对比
# ============================================================
print("\n" + "=" * 70)
print("📊 PER-CONDITION BREAKDOWN")
print("=" * 70)

cond_order = ["IDENTITY_RACE", "IDENTITY_GENDER", "CONTROL_SAMEGRP", "CONTROL_NONID"]
cond_labels = ["Identity\n(Race)", "Identity\n(Gender)", "Control\n(Same-Group)", "Control\n(Non-Identity)"]

print(f"\n  {'Condition':>25} {'N pairs':>8} {'Mean SS':>10} {'Func-TCD':>10} {'Cont-TCD':>10} {'F/C Ratio':>10}")
print(f"  {'─'*25} {'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")



cond_summary = {}
for cond in cond_order:
    results = all_results[cond]
    n = len(results)
    ss_mean = np.mean([r["ss"] for r in results])
    func_mean = np.mean([r["func_tcd_mean"] for r in results])
    cont_mean = np.mean([r["cont_tcd_mean"] for r in results])
    ratio = func_mean / cont_mean if cont_mean > 0 else float('inf')
    cond_summary[cond] = {"n": n, "ss": ss_mean, "func": func_mean, "cont": cont_mean, "ratio": ratio}
    print(f"  {cond:>25} {n:>8} {ss_mean:>10.6f} {func_mean:>10.6f} {cont_mean:>10.6f} {ratio:>10.2f}x")

# ============================================================
# 所有两两对比的统计检验
# ============================================================
print("\n" + "=" * 70)
print("📊 PAIRWISE COMPARISONS (Mann-Whitney U on function-word |TCD|)")
print("=" * 70)

pairwise_results = {}
for i, c1 in enumerate(cond_order):
    for j, c2 in enumerate(cond_order):
        if i >= j:
            continue
        d1 = condition_func_tcds[c1]
        d2 = condition_func_tcds[c2]
        u, p = stats.mannwhitneyu(d1, d2, alternative='two-sided')
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ratio = np.mean(d1) / np.mean(d2) if np.mean(d2) > 0 else float('inf')
        pairwise_results[f"{c1} vs {c2}"] = {"u": u, "p": p, "ratio": ratio}
        print(f"  {c1:>20} vs {c2:<20}: p={p:.8f} {sig:>4}  ratio={ratio:.3f}x")

# ============================================================
# 可视化
# ============================================================
print("\n📈 Generating visualizations...")

fig, axes = plt.subplots(1, 3, figsize=(21, 6))
fig.suptitle("P0: Is Function-Word Bias Identity-Specific?", fontsize=14, fontweight='bold')

# 1. Box plot per condition
box_data = [condition_func_tcds[c] for c in cond_order]
bp = axes[0].boxplot(box_data, labels=cond_labels, patch_artist=True, widths=0.6,
                      showmeans=True, meanprops=dict(marker='D', markerfacecolor='white', markersize=6))
colors = ['#e74c3c', '#c0392b', '#3498db', '#2980b9']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[0].set_ylabel("Function-Word |TCD|")
axes[0].set_title("Distribution of Func-Word |TCD| per Condition")
axes[0].grid(alpha=0.3)
# Add significance bracket
max_y = max(np.percentile(d, 95) for d in box_data)
axes[0].plot([1, 1, 2, 2], [max_y*1.05, max_y*1.1, max_y*1.1, max_y*1.05], 'k-', lw=1)
axes[0].text(1.5, max_y*1.12, f"Identity vs Control: p={p_func:.6f}", ha='center', fontsize=8,
             fontweight='bold' if p_func < 0.05 else 'normal')

# 2. Bar chart: Mean func-TCD and cont-TCD
x = np.arange(len(cond_order))
width = 0.35
func_vals = [cond_summary[c]["func"] for c in cond_order]
cont_vals = [cond_summary[c]["cont"] for c in cond_order]
axes[1].bar(x - width/2, func_vals, width, label='Function Words', color='#e74c3c', alpha=0.7)
axes[1].bar(x + width/2, cont_vals, width, label='Content Words', color='#3498db', alpha=0.7)
axes[1].set_xticks(x)
axes[1].set_xticklabels(cond_labels, fontsize=8)
axes[1].set_ylabel("Mean |TCD|")
axes[1].set_title("Function vs Content Word TCD by Condition")
axes[1].legend()
axes[1].grid(alpha=0.3, axis='y')

# 3. SS comparison
ss_vals = [cond_summary[c]["ss"] for c in cond_order]
bar_colors = ['#e74c3c', '#c0392b', '#3498db', '#2980b9']
axes[2].bar(range(len(cond_order)), ss_vals, color=bar_colors, alpha=0.7, edgecolor='white')
axes[2].set_xticks(range(len(cond_order)))
axes[2].set_xticklabels(cond_labels, fontsize=8)
axes[2].set_ylabel("Mean Score Sensitivity (SS)")
axes[2].set_title("Overall Bias (SS) by Condition")
axes[2].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "p0_identity_vs_control.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ p0_identity_vs_control.png")

# ============================================================
# 关键衍生分析: Function-word amplification ratio per condition
# ============================================================
print("\n📈 Generating F/C ratio chart...")

fig, ax = plt.subplots(figsize=(8, 5))
ratios = [cond_summary[c]["ratio"] for c in cond_order]
bar_colors = ['#e74c3c', '#c0392b', '#3498db', '#2980b9']
bars = ax.bar(range(len(cond_order)), ratios, color=bar_colors, alpha=0.7, edgecolor='white')
ax.axhline(y=1.0, color='black', linewidth=1, linestyle='--', label='Ratio = 1 (no amplification)')
ax.set_xticks(range(len(cond_order)))
ax.set_xticklabels(cond_labels, fontsize=9)
ax.set_ylabel("Function-Word / Content-Word TCD Ratio")
ax.set_title("Is Function-Word Amplification Specific to Identity Swaps?", fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='y')

for i, (r, bar) in enumerate(zip(ratios, bars)):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{r:.2f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "p0_fc_ratio.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ p0_fc_ratio.png")

# ============================================================
# 保存结果
# ============================================================
output = {
    "summary": {
        "identity_func_mean": float(np.mean(identity_func)),
        "control_func_mean": float(np.mean(control_func)),
        "ratio": float(ratio_func),
        "mannwhitney_p": float(p_func),
        "cohens_d": float(d_func),
    },
    "per_condition": {c: cond_summary[c] for c in cond_order},
    "pairwise": pairwise_results,
}

with open(OUTPUT_DIR / "p0_results.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

# ============================================================
# 最终判定
# ============================================================
print("\n" + "=" * 70)
print("🏁 P0 VERDICT")
print("=" * 70)

if p_func < 0.05 and ratio_func > 1.2:
    print(f"""
  ✅ P0 PASS: Function-word TCD effect IS identity-specific!

  Identity swap func-TCD ({np.mean(identity_func):.6f}) is significantly
  greater than control swap func-TCD ({np.mean(control_func):.6f}).
    ratio = {ratio_func:.3f}x
    p = {p_func:.8f}
    Cohen's d = {d_func:.4f}

  → The function-word bias propagation phenomenon is NOT just generic
    ColBERT sensitivity. It is amplified when identity markers are changed.
  → Safe to proceed to repo creation and Phase 3.
""")
elif p_func < 0.05:
    print(f"""
  ⚠️ P0 PARTIAL PASS: Effect is statistically significant but small.

  Identity func-TCD ({np.mean(identity_func):.6f}) > Control ({np.mean(control_func):.6f})
    ratio = {ratio_func:.3f}x (below 1.2x threshold)
    p = {p_func:.8f}

  → The effect exists but is modest. The paper narrative should be:
    "identity swaps cause SOMEWHAT more function-word disruption than
     ordinary swaps, but the baseline sensitivity is also notable."
""")
else:
    print(f"""
  ❌ P0 FAIL: Function-word TCD effect is NOT identity-specific.

  Identity func-TCD ({np.mean(identity_func):.6f}) vs Control ({np.mean(control_func):.6f})
    ratio = {ratio_func:.3f}x
    p = {p_func:.8f} (not significant)

  → The function-word volatility is a GENERAL property of ColBERT,
    not specific to identity swaps. The paper narrative needs revision:
    focus on SS (total bias) as the identity-specific signal, and
    frame function-word TCD as a ColBERT architectural property.
""")

print("✨ P0 experiment complete!")
