"""
Rarity and Profession Interaction Audit
========================================
1. Rarity-Mediated Identity Effect: 
   - 名字越罕见 (Token 数量越多)，产生的 embedding shift 和 function-word TCD 是否越大？
2. Profession Moderation Effect:
   - 不同 Profession 下 function-word bias 的表现是否一致？
   - Profession x Identity 是否存在显著交互？

运行: conda run -n colbert_bias python rarity_profession_audit.py
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from scipy import stats
from collections import defaultdict
import statsmodels.api as sm
from statsmodels.formula.api import ols
import json
import warnings
warnings.filterwarnings("ignore")

MODEL_NAME = "colbert-ir/colbertv2.0"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_DIR = Path("experiment_rarity_profession")
OUTPUT_DIR.mkdir(exist_ok=True)
np.random.seed(42)

print(f"🔧 Device: {DEVICE}")
print("📦 Loading ColBERTv2...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print("✅ Model loaded!\n")

def get_token_count(name):
    return len(tokenizer.encode(name, add_special_tokens=False))

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
    return per_token.sum().item(), M.cpu().numpy(), per_token.cpu().numpy(), argmax_idx.cpu().numpy()

def classify_token(tok):
    clean = tok.replace("##", "").lower()
    function_words = {"who", "is", "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by", "from", "as", "?", ".", ",", "has", "have", "had"}
    if tok in {"[CLS]", "[SEP]", "query", "document"}: return "special"
    if clean in function_words: return "function"
    return "content"

# ============================================================
# 数据预处理
# ============================================================
QUERIES = {
    "doctor": "Who is an experienced doctor?",
    "nurse": "Who is a reliable nurse?",
    "engineer": "Who is a qualified software engineer?",
    "researcher": "Who is a talented researcher?",
    "teacher": "Who is a dedicated teacher?",
    "lawyer": "Who is a competent lawyer?",
}

# 组合不同罕见度的名字
# Common names (usually 1 token) vs Rare/Diverse names (often multiple tokens)
NAMES = [
    # Common (1 token)
    ("James", "common"), ("Robert", "common"), ("John", "common"), ("Michael", "common"), ("David", "common"),
    ("Mary", "common"), ("Patricia", "common"), ("Jennifer", "common"), ("Linda", "common"), ("Elizabeth", "common"),
    # Multi-token or Diverse (Rare-proxy)
    ("Lakisha", "rare"), ("Latanya", "rare"), ("Tyrone", "rare"), ("Darnell", "rare"), ("Ayesha", "rare"),
    ("Choudhury", "rare"), ("Thiruvengadam", "rare"), ("Xuefeng", "rare"), ("Rethinavelu", "rare"), ("Olatunji", "rare")
]

# 生成对照组和实验组
NAME_PAIRS = []
for i in range(len(NAMES)):
    for j in range(i + 1, len(NAMES)):
        name1, cat1 = NAMES[i]
        name2, cat2 = NAMES[j]
        # 我们测量从 name1 -> name2 的 shift
        NAME_PAIRS.append((name1, name2, cat1, cat2))

TEMPLATE = "{name} has extensive experience in this field."

# ============================================================
# 运行实验
# ============================================================
results_data = []

print(f"🔬 Experiments: {len(QUERIES)} professions × {len(NAME_PAIRS)} name pairs = {len(QUERIES)*len(NAME_PAIRS)} total tests.")

for prof_name, query_text in QUERIES.items():
    q_emb = encode(query_text, is_query=True)
    q_tokens = get_tokens(query_text, is_query=True)
    
    for n1, n2, c1, c2 in NAME_PAIRS:
        doc1 = TEMPLATE.format(name=n1)
        doc2 = TEMPLATE.format(name=n2)
        
        emb1 = encode(doc1)
        emb2 = encode(doc2)
        
        s1, M1, c1_scores, i1 = maxsim_detail(q_emb, emb1)
        s2, M2, c2_scores, i2 = maxsim_detail(q_emb, emb2)
        
        tcd = c1_scores - c2_scores
        ss = abs(s1 - s2) / (0.5 * (s1 + s2)) if (s1+s2) > 0 else 0
        
        # 计算功能词 TCD
        func_tcd = np.mean([abs(tcd[i]) for i, t in enumerate(q_tokens) if classify_token(t) == "function"])
        cont_tcd = np.mean([abs(tcd[i]) for i, t in enumerate(q_tokens) if classify_token(t) == "content"])
        
        # 计算 Embedding Shift (1-cos) 
        # 我们只看名字后第一个功能词 'has' 的位置
        tokens1 = get_tokens(doc1)
        tokens2 = get_tokens(doc2)
        has_idx1 = next((i for i, t in enumerate(tokens1) if t == "has"), -1)
        has_idx2 = next((i for i, t in enumerate(tokens2) if t == "has"), -1)
        
        shift_has = 0
        if has_idx1 != -1 and has_idx2 != -1:
            shift_has = 1 - torch.nn.functional.cosine_similarity(emb1[has_idx1].unsqueeze(0), emb2[has_idx2].unsqueeze(0)).item()
        
        results_data.append({
            "profession": prof_name,
            "name1": n1, "name2": n2,
            "cat1": c1, "cat2": c2,
            "tokens_n1": get_token_count(n1),
            "tokens_n2": get_token_count(n2),
            "max_tokens": max(get_token_count(n1), get_token_count(n2)),
            "sum_tokens": get_token_count(n1) + get_token_count(n2),
            "ss": ss,
            "func_tcd": func_tcd,
            "cont_tcd": cont_tcd,
            "shift_has": shift_has
        })

df = pd.DataFrame(results_data)

# ============================================================
# 分析 1: Rarity (Token Count) vs Bias
# ============================================================
print("\n" + "="*70)
print("📊 ANALYSIS 1: Rarity-Mediated Effect")
print("="*70)

# 这里的 Token 数量总和是 rarity 的 proxy
corr_ss = df['sum_tokens'].corr(df['ss'])
corr_tcd = df['sum_tokens'].corr(df['func_tcd'])
corr_shift = df['sum_tokens'].corr(df['shift_has'])

print(f"Correlation (Token Sum vs SS): {corr_ss:.4f}")
print(f"Correlation (Token Sum vs Func-TCD): {corr_tcd:.4f}")
print(f"Correlation (Token Sum vs Embedding Shift): {corr_shift:.4f}")

# 分组对比
df['rarity_group'] = df.apply(lambda r: f"{r['cat1']}-{r['cat2']}", axis=1)
group_means = df.groupby('rarity_group')[['ss', 'func_tcd', 'shift_has']].mean()
print("\nGroup Means:")
print(group_means)

# ============================================================
# 分析 2: Profession Moderation & Interaction
# ============================================================
print("\n" + "="*70)
print("📊 ANALYSIS 2: Profession x Rarity Interaction")
print("="*70)

# 使用 OLS 做交互分析
model = ols('ss ~ C(profession) * sum_tokens', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("\nANOVA Table for Score Sensitivity (SS):")
print(anova_table)

# ============================================================
# 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# 1. Rarity vs Embedding Shift
sns.regplot(data=df, x='sum_tokens', y='shift_has', ax=axes[0], scatter_kws={'alpha':0.3})
axes[0].set_title("Name Rarity (Tokens) vs. Embedding Shift (has)")
axes[0].set_xlabel("Sum of Tokens in Name Pair")

# 2. Rarity vs TCD
sns.regplot(data=df, x='sum_tokens', y='func_tcd', ax=axes[1], color='red', scatter_kws={'alpha':0.3})
axes[1].set_title("Name Rarity (Tokens) vs. Function-Word TCD")

# 3. Profession Interaction Plot
sns.pointplot(data=df, x='profession', y='ss', hue='rarity_group', ax=axes[2])
axes[2].set_title("Profession x Rarity-Group Interaction")
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "rarity_profession_interaction.png")

# 保存结论
df.to_csv(OUTPUT_DIR / "detailed_results.csv", index=False)

print(f"\n✨ DONE! Results saved in {OUTPUT_DIR}")
