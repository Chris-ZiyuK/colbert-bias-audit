# ColBERT Bias Audit — 项目完整技术文档

> **Where Does Bias Hide? Token-Level Attribution Reveals Distributed Bias Propagation in Late-Interaction Retrieval**
>
> Chris Ziyu Kong · Brown University · CSCI 2952W Critical AI and Data Studies · 2026

本文档是对本研究项目从理论基础、技术方法论、实验设计、代码实现、统计结果到当前未解决问题的**完整、可独立理解的技术复盘**。任何人（包括未来的 AI Agent、合作研究者、PI）仅凭此文档即可全面还原项目当前状态。

---

## 目录

1. [理论基础与研究动机](#1-理论基础与研究动机)
2. [技术背景：ColBERT 与 Late Interaction 架构](#2-技术背景colbert-与-late-interaction-架构)
3. [核心方法论：TCD 指标体系](#3-核心方法论tcd-指标体系)
4. [实验数据集与名字矩阵设计](#4-实验数据集与名字矩阵设计)
5. [实验阶段详解](#5-实验阶段详解)
   - [5.1 Phase 1：偏见存在性验证与功能词发现](#51-phase-1偏见存在性验证与功能词发现)
   - [5.2 P0：身份特异性对照实验](#52-p0身份特异性对照实验)
   - [5.3 P1：33 职业系统化扩展](#53-p133-职业系统化扩展)
   - [5.4 P2：名字混淆因素拆解（43,065 次测试）](#54-p2名字混淆因素拆解43065-次测试)
   - [5.5 P2b：Rosenman 靶向验证](#55-p2brosenman-靶向验证)
   - [5.6 RT：真实文本生态效度验证](#56-rt真实文本生态效度验证)
   - [5.7 BM25：非上下文模型基线对照](#57-bm25非上下文模型基线对照)
6. [核心发现汇总与可引用结论](#6-核心发现汇总与可引用结论)
7. [代码架构与文件索引](#7-代码架构与文件索引)
8. [论文状态与投稿目标](#8-论文状态与投稿目标)
9. [已知局限与未解决问题](#9-已知局限与未解决问题)
10. [下一步行动清单](#10-下一步行动清单)
11. [AI Agent 决策指南](#11-ai-agent-决策指南)

---

## 1. 理论基础与研究动机

### 1.1 核心问题

**搜索引擎会不会因为你的名字而歧视你？**

当一个密集检索模型（如 ColBERT）对两篇完全相同的简历打分时，仅仅因为名字从 "Emily" 变成了 "Lakisha"，分数是否会发生显著变化？如果会，偏见到底藏在模型的哪个位置？

### 1.2 理论来源

本项目的灵感交汇于两个领域：

- **社会学 / 数据研究**：Bowker & Star (1999) 的*分类的政治学* —— 分类系统从来不是中性的，算法分类中嵌入了社会价值判断。
- **劳动经济学**：Bertrand & Mullainathan (2004) 的简历审计实验 —— 投递完全相同的简历，仅将名字在 "Emily"（白人刻板印象名）和 "Lakisha"（黑人刻板印象名）之间切换，前者的回调率高出 50%。

### 1.3 论文定位

本项目被严格定位为**诊断/测量型论文 (Measurement / Audit Paper)**，而非去偏算法论文 (Debiasing Paper)。

核心叙事：ColBERT 的 Late Interaction 架构赋予了我们一台 **"X 光机"**，让我们首次能够深入密集检索模型内部，精准拆解偏见的**分布位置**与**传播路径**。我们不只是说"偏见存在"，而是说"偏见藏在这里、通过这种方式传播"。

### 1.4 论文核心贡献点（按 PI 反馈重聚焦）

| # | 类型 | 内容 | 优先级 |
|---|------|------|--------|
| C1 | 方法论 | 提出 TCD (Token Contribution Disparity) 指标，实现 Token 级偏见归因 | 🟢 核心 |
| C2 | 实证发现 | 揭示功能词偏见传导机制（分布式传播） | 🟢 核心 |
| C3 | 实证发现 | 发现罕见度中介效应 + 名字频率/SES 混淆因素拆分 | 🟢 核心 |
| C4 | 验证 | P0 对照证明效应的身份特异性 + 系统化职业分析 | 🟢 核心 |
| C5 | 泛化 | 真实文本生态效度检验 + BM25 对照 | 🟢 核心（已完成） |
| C6 | 应用 | Null-Space Projection 功能词去偏 | 🟡 Stretch Goal |

---

## 2. 技术背景：ColBERT 与 Late Interaction 架构

### 2.1 Dense Retrieval 模型谱系

| 模型 | 架构类型 | 表示粒度 | 可解释性 |
|------|----------|----------|----------|
| **BM25** | 传统词频统计 | 词级 | 完全可解释（TF-IDF） |
| **DPR** | Bi-encoder / 单向量 | 整个序列压缩为一个向量 | 黑盒 —— 无法分解到词级 |
| **SPLADE** | 学习型稀疏权重 | 词级（稀疏） | 部分可解释 |
| **ColBERT** | Late Interaction / 多向量 | **保留每个 Token 的独立向量** | **MaxSim 可完全分解到词级** |

### 2.2 ColBERT 的 MaxSim 打分机制

ColBERTv2 (Santhanam et al., 2022) 将 Query 和 Document 分别编码为 L2 归一化的 per-token embedding 序列，然后通过 **MaxSim** 计算相关性分数：

```
S(Q, D) = Σ_{i=1}^{|Q|} max_{j ∈ [1,|D|]} q_i^T · d_j
```

其中 `q_i` 是 Query 第 i 个 Token 的 embedding，`d_j` 是 Document 第 j 个 Token 的 embedding。

**关键洞察**：总分是所有 Query Token 贡献的**简单求和**。这意味着我们可以精确测量**每一个 Query Token 对总分贡献了多少差异** —— 这在 DPR 等单向量模型中是不可能的。

### 2.3 为什么只有 ColBERT 能当 "X 光机"

- **DPR**：整个文档压缩成单个 768 维向量，偏见被彻底混合在一起，无法提问"是哪个词导致了分数差异"。
- **ColBERT**：保留了所有 Token 的独立向量，MaxSim 的逐词求和结构使得每个 Token 的打分贡献可以被精确提取。
- **类比**：X 光机不会让你生病，但它让你看到病灶在哪里。ColBERT 不一定比其他模型更有偏见，但它是唯一能让你**看到偏见内部结构**的模型。

---

## 3. 核心方法论：TCD 指标体系

### 3.0 检索场景与实验设置（关键！）

#### 模拟的场景

我们模拟的是 **候选人检索 (Candidate Retrieval)** 场景：雇主在搜索引擎中输入职业需求查询，系统从候选人文档库中检索并排序匹配的候选人描述。

#### Query 和 Document 分别是什么

| 角色 | 内容 | 编码前缀 | 示例 |
|------|------|---------|------|
| **Query** | 职业搜索查询 | `"query: "` | `"query: Who is a qualified doctor?"` |
| **Document** | 候选人描述（含身份标记） | `"document: "` | `"document: Emily has over ten years of clinical experience."` |

- Query 来自 `data/professions/professions.json` 中的 33 个标准职业查询
- Document 由**模板 + 名字/代词**组成，模板包含职业相关的资历描述
- **身份替换发生在 Document 端**：`D_A` 含 "Emily"，`D_B` 含 "Lakisha"
- **Query 保持不变**：同一个 "Who is a qualified doctor?" 用于对比两篇文档

#### 完整的实验流程

```
对于每组 (profession, name_A, name_B, template):

  1. 编码 Query：
     q_emb = encode("query: Who is a qualified doctor?")
     → 产生 per-token query embedding 矩阵 [|Q| × 128]

  2. 构造反事实文档对：
     doc_A = "Emily has over ten years of experience in this field."
     doc_B = "Lakisha has over ten years of experience in this field."

  3. 分别编码两篇 Document：
     d_emb_A = encode("document: Emily has over ten years ...")
     d_emb_B = encode("document: Lakisha has over ten years ...")
     → 每篇产生 per-token document embedding 矩阵 [|D| × 128]

  4. 计算 MaxSim 分数：
     S(Q, D_A) = Σᵢ maxⱼ (qᵢᵀ · dⱼᴬ)  → e.g., 78.3
     S(Q, D_B) = Σᵢ maxⱼ (qᵢᵀ · dⱼᴮ)  → e.g., 75.9

  5. 计算 per-token TCD：
     对每个 query token qᵢ:
       cᵢᴬ = maxⱼ (qᵢᵀ · dⱼᴬ)    // qᵢ 对 doc_A 的贡献
       cᵢᴮ = maxⱼ (qᵢᵀ · dⱼᴮ)    // qᵢ 对 doc_B 的贡献
       TCD(qᵢ) = |cᵢᴬ − cᵢᴮ|     // 偏见的 token 级归因

  6. 聚合为 Func-TCD, Cont-TCD, SS
```

#### 为什么 Pairwise 比较等价于 Full-Index 检索

ColBERT 架构的一个关键属性：**`S(Q, D)` 的计算对每个 (Q, D) 对是完全独立的。**

- Document 编码时**不知道 Query 是什么**（离线编码阶段独立进行）
- Document 编码时**不知道其他 Document 是什么**（每篇 Document 独立编码）
- `S(Q, D_Emily)` 的值**完全不依赖**于 `D_Lakisha` 是否在索引中

因此：

```
S(Q, D_Emily) 在我们的 pairwise 实验中
    ===（数学上严格等价）===
S(Q, D_Emily) 在包含 8.8M 篇文档的 MS MARCO 索引中
```

这个独立性属性是所有 bi-encoder 和 late-interaction 架构（包括 DPR、ColBERT、SPLADE）共有的。它保证了我们的 pairwise 比较忠实反映了模型在生产检索流水线中的打分行为。

> **注意**：我们目前**没有测试**的是 ranking impact —— 即这些分数差异在一个包含大量候选文档的真实检索池中是否足以改变排名位置。这是一个独立的后续实验（见 §10 下一步行动 E1）。

### 3.1 反事实对 (Counterfactual Pair) 构造

一个反事实对 `(D_A, D_B)` 由两篇完全相同的文档组成，**唯一的区别**是身份标记（名字或代词）：

```
D_A: "Emily has over ten years of experience in this field."
D_B: "Lakisha has over ten years of experience in this field."
```

### 3.2 TCD (Token Contribution Disparity) 定义

对于 Query Token `q_i`，它对两篇文档的 MaxSim 贡献分别为：

```
c_i^A = max_j (q_i^T · d_j^A)
c_i^B = max_j (q_i^T · d_j^B)
```

**TCD** 定义为两者之差的绝对值：

```
TCD(q_i) = |c_i^A − c_i^B|
```

TCD 值高意味着：当文档中的身份标记被替换时，这个 Query Token 的打分贡献发生了显著变化 —— 身份信息"泄漏"到了这个 Token 的上下文表示中。

### 3.3 聚合指标

我们将 Query Token 分为两类：
- **功能词 (Function words)**：`who`, `is`, `a`, `an`, `the`, `?`, `has`, `are`, `in`, `on`, `to`, `for`, `of`, `with` 等（完整列表见 `src/audit/core.py` L15-22）
- **内容词 (Content words)**：`doctor`, `qualified`, `talented`, `nurse`, `experienced` 等

然后计算：

```
Func-TCD = (1/|F|) Σ_{q_i ∈ F} TCD(q_i)    // 功能词平均 TCD
Cont-TCD = (1/|C|) Σ_{q_i ∈ C} TCD(q_i)    // 内容词平均 TCD
```

**Score Sensitivity (SS)** —— 归一化的总分偏差：

```
SS = |S(Q, D_A) − S(Q, D_B)| / (½ × (S(Q, D_A) + S(Q, D_B)))
```

### 3.4 核心代码实现

所有度量的实现位于 `src/audit/core.py`（126 行）：

- `encode(text, tokenizer, model, device, is_query)` —— 文本编码为 L2 归一化的 ColBERT embedding
- `maxsim_detail(q_emb, d_emb)` —— 计算 MaxSim 总分 + per-token 贡献 + argmax 索引
- `classify_token(tok)` —— 将 Token 分类为 `function` / `content` / `special`
- `compute_tcd_breakdown(q_emb, d_emb_a, d_emb_b, q_tokens)` —— 一次调用返回 SS、Func-TCD、Cont-TCD、TCD ratio、raw TCD array

模型使用 `colbert-ir/colbertv2.0`（HuggingFace），自动选择 CUDA > MPS > CPU。

---

## 4. 实验数据集与名字矩阵设计

### 4.1 名字池（初始 30 名 → P2b 扩展至 65 名）

初始名字池 (`data/audit_names/name_features.json`) 包含 30 个名字，每个名字标注了：

| 属性 | 说明 | 示例 |
|------|------|------|
| `race` | 4 类：White, Black, Hispanic, Asian | Emily=White, Lakisha=Black |
| `gender` | F / M | |
| `census_freq_per_100k` | 美国人口普查中的频率（每 10 万） | Emily=844, Lakisha=5, Thiruvengadam=0.5 |
| `bpe_tokens` | ColBERTv2 tokenizer 实际切分的 Token 数 | Emily=1, Lakisha=3, Thiruvengadam=5 |
| `ses_tier` | 社会经济地位：high / mid / low | Emily=high, Lakisha=low |

**被精心设计的"混淆因素打破者"名字**：
- `Wei` (Asian, M)：频率=12，但**只有 1 个 token** —— 打破"罕见名=多 token"假设
- `Mei` (Asian, F)：频率=8，1 token —— 同上
- `Maria` (Hispanic, F)：频率=577，1 token —— 打破"少数族裔名=低频"假设
- `Jasmine` (Black, F)：频率=215，2 token —— 打破"黑人名=极低频"假设

### 4.2 P2b Rosenman 扩展矩阵

在 P2b 阶段，我们从 Rosenman, Olivella & Imai (2023) 的种族-名字概率数据集（Harvard Dataverse DOI: `10.7910/DVN/SGKW0K`，基于 6 个南方州的选民档案）中提取了更多名字，将名字池扩展到 65 个。

扩展后的**匹配对 (Matched-pair Family)** 设计：

| 对照组 | 固定变量 | 变化变量 | 对数 (旧→新) |
|--------|----------|----------|-------------|
| **Race** | gender, token count, frequency | race | 35→59 |
| **Gender** | race, token count, frequency | gender | 27→44 |
| **Tokenization** | race, gender, frequency | BPE token count | 17→49 |
| **Frequency gap** | race, gender, token count | log-frequency gap | 35→43 |
| **Absolute rarity** | race, gender, token count, ~frequency | mean log-frequency | 4→5 |

### 4.3 职业集（33 个职业 × 7 个 BLS 类别）

`data/professions/professions.json` 包含 33 个职业，每个标注了：
- `query`：标准查询格式（如 "Who is an experienced doctor?"）
- `bls_category`：BLS 职业大类（healthcare, STEM, education, white-collar, leadership, blue-collar, service）
- `prestige`：high / mid / low
- `bls_female_pct`：该职业中女性占比（美国劳工统计局数据）

完整职业列表：doctor, nurse, surgeon, therapist, dentist, software engineer, researcher, data scientist, civil engineer, biologist, teacher, professor, librarian, lawyer, accountant, financial analyst, consultant, CEO, manager, director, electrician, plumber, mechanic, construction worker, truck driver, welder, secretary, social worker, receptionist, janitor, chef, pilot, firefighter。

### 4.4 文档模板

大多数实验使用 3-5 个模板，格式为 `{name} has extensive experience in this field.` 或 `{pron} has over ten years of experience...`。P2 使用 3 个模板以确保结果不依赖于特定句式。

---

## 5. 实验阶段详解

### 5.1 Phase 1：偏见存在性验证与功能词发现

#### 5.1.1 初步验证（783 对反事实测试）

**脚本**：`experiments/phase1_synthetic/targeted_experiments.py`（709 行）

**设计**：
- Study 1：系统化代词偏见检测（He / She / They × 12 职业 × 4 模板）
- Study 2：职业特异性偏见（7 职业 × 21 名字/代词对 × 5 模板 = 783 对）

**统计方法**：
- 配对 t 检验 + Bonferroni 校正（α = 0.05/3 = 0.0167）
- Wilcoxon 符号秩检验（非参数替代）
- 单因素 ANOVA + Friedman χ² 检验
- Cohen's d 效应量 + Bootstrap 95% 置信区间
- 二项检验（方向性分析）
- Kruskal-Wallis H 检验（跨职业比较）

**关键结果**：
- 偏见确实存在（p < 0.001）
- "They" 代词被系统性降分：Mean Δ(Male−NB) 在所有 12 职业中一致为正
- 职业特异性：nurse 和 researcher 的 SS 显著高于 doctor 和 electrician

#### 5.1.2 功能词偏见发现（650 对大规模验证）

**脚本**：`experiments/phase1_synthetic/function_word_bias.py`（520 行）

**设计**：10 职业 × (10 名字对 + 3 代词对) × 5 模板 = 650 反事实对

**四层分析**：

**Analysis 1 —— 功能词 vs 内容词核心对比**：

|  | N | Mean \|TCD\| | Median | Ratio |
|---|---|---|---|---|
| **Function words** | 2,535 | **0.0210** | 0.0133 | **1.44×** |
| Content words | 4,225 | 0.0146 | 0.0088 | — |

Mann-Whitney U test (function > content): **p = 0.0002 \*\*\***

**Analysis 2 —— 逐 Token 排名**：每个具体 Token 的 Mean |TCD|、方向性（%→A）和 t 检验 p 值。例如：
- `is`：72.3% 的时间偏向多数群体（p < 0.0001 \*\*\*）
- `?`：68.6%（p < 0.0001 \*\*\*）
- `a`：64.3%（p < 0.0001 \*\*\*）
- `who`：59.1%（p = 0.00001 \*\*\*）

**Analysis 3 —— 跨职业一致性**：10 个职业中 9 个呈现功能词 > 内容词（最高 experienced doctor 2.24×，唯一例外 skilled electrician 0.89×）。

**Analysis 4 —— 嵌入扰动机制分析**：直接测量身份替换后每个 Token 的 128 维 embedding 向量变化（cosine distance）：

| Token 类型 | Mean Embedding Shift | N |
|---|---|---|
| **Function words** | **0.1137** | 9 |
| Content words | 0.0128 | 6 |

Function word embeddings shift **8.9×** more than content word embeddings (p = 0.0002 \*\*\*)。

**机制解释**（关键因果链）：

```
身份替换 (Emily → Lakisha)
      │
      ▼
BERT Self-Attention 重新计算所有 Token 的 embedding
      │
      ├── 内容词 ("clinical", "experience"): 偏移小 — 有强语义锚点
      │
      └── 功能词 ("has", "is", "a"): 偏移大 — 无语义锚点，纯粹的"上下文海绵"
              │
              ▼
Query 功能词 ("who", "is", "a") 找到不同的 MaxSim 匹配
              │
              ▼
TCD 升高 → 总分偏差
```

#### 5.1.3 罕见度中介效应（1,140 组测试）

**脚本**：`experiments/phase1_synthetic/rarity_profession_audit.py`（8,621 字节）

**设计**：6 职业 × 190 唯一名字对 = 1,140 组

**关键结果**：

| 名字组合 | Mean SS（总偏见） | Mean Func-TCD | 倍率 |
|----------|---|---|---|
| Common-Common | 0.0210 | 0.0108 | 1.0× |
| Common-Rare | 0.0305 | 0.0163 | 1.5× |
| **Rare-Rare** | **0.0366** | **0.0199** | **1.84×** |

**Two-Way ANOVA**：
- Profession: F = 214.28, p < 0.0001 \*\*\*
- Name Rarity (Sum of Tokens): F = 300.99, p < 0.0001 \*\*\*
- Interaction: F = 1.13, p = 0.3421 (n.s.)

**机制**：罕见名字被 BPE tokenizer 切碎成多个 subword token（如 "Thiruvengadam" → "Th", "##iru", "##ven", "##gada", "##m"），增加模型不确定性，导致更依赖训练数据中的刻板印象来"填空"。

**相关系数**：
- Token 数 vs Function-Word TCD: r = 0.3801（强正相关）
- Token 数 vs Total Bias (SS): r = 0.3134

### 5.2 P0：身份特异性对照实验

**脚本**：`experiments/phase0_control/p0_control_experiment.py`（432 行）

**核心问题**：功能词的高 TCD 是身份替换的特异性效应，还是 ColBERT 对**任何**词汇替换都一样敏感？

**设计**：4 组对照条件 × 6 Query × 5 模板 = 1,080 组

| 条件 | 含义 | 替换示例 | Mean Func-TCD | 与基线比率 |
|------|------|---------|---|---|
| **IDENTITY_GENDER** | 跨性别代词替换 | He → They | **0.0298** | **1.92×** |
| IDENTITY_RACE | 跨种族名字替换 | Emily → Lakisha | 0.0192 | 1.24× |
| CONTROL_SAMEGRP | 同人口学组内替换 | Emily → Jennifer | 0.0177 | 1.14× |
| CONTROL_NONID | 非身份词替换 | ten → fifteen | 0.0155 | 1.0× (基线) |

**核心统计**：
- 合并身份组 vs 合并对照组：Mann-Whitney U, **p < 1e-8**, Cohen's **d = 0.37**
- **结论**：P0 PASS ✅ —— 功能词偏见传导不只是通用敏感性，它在跨越身份边界时被**进一步放大**。

### 5.3 P1：33 职业系统化扩展

**脚本**：
- `experiments/phase2_profession_expansion/profession_expansion.py`（14,892 字节）
- `experiments/phase2_profession_expansion/p1_wrapup_analysis.py`（12,819 字节）

**设计**：33 职业 × (10 name pairs + 3 pronoun swaps) × 3 模板 = 1,287 测试

**结果**（来自 `results/p1_profession_expansion/p1_wrapup_stats.txt`）：

| 指标 | Name Swap | Pronoun Swap |
|------|-----------|-------------|
| N | 990 | 297 |
| Mean SS | 0.033327 | 0.034976 |
| Mean Func-TCD | 0.017705 | 0.020940 |
| Mean Cont-TCD | 0.013339 | 0.013549 |

**关键统计**：
- **Overall function > content**: Wilcoxon p = **2.2e-47** \*\*\*
- **Name swap**: Func > Cont, p = 7.55e-30 \*\*\*
- **Pronoun swap**: Func > Cont, p = 1.71e-21 \*\*\*
- **Pronoun > Name (Func-TCD)**: p = 0.00348 \*\* —— 代词替换在 Token 级别引发更强的功能词扰动
- **Pronoun > Name (Func−Cont gap)**: p = 0.0169 \*
- **跨职业类别差异**: Kruskal-Wallis H 检验不显著（name: p=0.90, pronoun: p=0.56），说明效应在各类职业中均匀存在

**职业级别**：33 个职业中 **30 个**呈现功能词 > 内容词的 TCD 模式。
**高偏见职业**：nurse, janitor, plumber, secretary。

### 5.4 P2：名字混淆因素拆解（43,065 次测试）

**脚本**：`experiments/phase2_name_confound/name_confound_decomposition.py`（567 行）

**核心问题**：经典的名字替换实验（Emily → Lakisha）到底在测量什么？是种族、性别、名字频率、BPE token 数，还是这些因素的混合物？

**设计**：
- 30 名字（所有组合 C(30,2) = 435 对）× 33 职业 × 3 模板 = **43,065 次测试**
- 回归模型：OLS with pair-level cluster-robust standard errors + profession & template fixed effects

**主回归公式**：
```
SS (or func_tcd) ~ cross_race + cross_gender + token_diff + mean_tokens
                   + freq_diff + mean_log_freq + C(ses_pair)
                   + C(profession) + C(template_id)
```

**嵌套模型阶梯**（来自 `results/p2_name_confound/p2_final_report.txt`）：

SS 模型：
| 模型 | R² | cross_race p | cross_gender p | freq_diff p |
|------|----|-------------|---------------|------------|
| m0 (仅 race) | 0.157 | 0.463 | — | — |
| m1 (+gender) | 0.183 | 0.209 | **3.2e-16** | — |
| m2 (+rarity/token) | 0.185 | 0.331 | **5.1e-16** | 0.302 |
| m3 (+SES, full) | **0.203** | 0.400 | **4.9e-15** | **0.027** |

Func-TCD 模型：
| 模型 | R² | cross_race p | cross_gender p | freq_diff p |
|------|----|-------------|---------------|------------|
| m0 (仅 race) | 0.156 | **0.011** | — | — |
| m1 (+gender) | 0.184 | **0.001** | **1.8e-15** | — |
| m2 (+rarity/token) | 0.187 | 0.059 | **3.2e-15** | **0.013** |
| m3 (+SES, full) | **0.204** | 0.124 | **2.2e-14** | **0.0002** |

**关键解读**：
- `cross_race` 在简化模型中对 Func-TCD 显著，但随着加入更多控制变量后**逐步减弱**直至边缘或不显著
- `cross_gender` 在所有规格中都**极其稳定显著** (p < 1e-14)
- `freq_diff` 在完整模型中对 Func-TCD 显著，但在严格匹配子集中减弱

**模板鲁棒性**：cross_gender 在 3 个模板中全部显著（p < 1e-8 到 p < 1e-18），cross_race 仅在 1 个模板中显著。

**正交子集面板**：进一步用 race-isolated、gender-isolated、tokenization-isolated 子集验证，确认：
- Gender: 最稳
- Tokenization: 在严格匹配后重新显现
- Race: 仅保留弱的 token-level 残余信号

### 5.5 P2b：Rosenman 靶向验证

**脚本**：
- `experiments/phase2_name_confound/build_p2b_rosenman_matrix.py`（12,330 字节）—— 构建扩展矩阵
- `experiments/phase2_name_confound/p2b_rosenman_validation.py`（7,413 字节）—— 靶向验证

**设计**：使用扩展到 65 名字的 Rosenman 矩阵，针对 5 个匹配族分别运行 OLS 回归：
```
SS (or func_tcd) ~ treatment + C(profession) + C(template_id)
```

**最终结果**（`results/p2_name_confound/p2b_rosenman_report.txt`）：

| Factor | Pairs | SS coef | SS p | Func-TCD coef | Func-TCD p | R² |
|--------|-------|---------|------|---------------|------------|-----|
| **Gender** | 107 | +0.0088 | **6.2e-7** | **+0.0049** | **4.5e-9** | 0.215 |
| **Tokenization** | 103 | +0.0067 | **0.009** | **+0.0048** | **5.3e-4** | 0.116 |
| **Race** | 124 | +0.0043 | **0.014** | +0.0026 | **0.005** | 0.123 |
| Frequency gap | 100 | +0.0015 | 0.405 | +0.0008 | 0.383 | 0.112 |
| Absolute rarity | 28 | +0.0024 | 0.492 | +0.0032 | 0.084 | 0.142 |

**解读**：
- ✅ **Gender** 是最强信号（两个指标均 p < 1e-6）
- ✅ **Tokenization** 获得突破性验证（从先前不稳定到现在双显著）
- ✅ **Race** 在 Rosenman 扩展后再次可检测到（但比 gender 和 tokenization 弱）
- ❌ **Frequency gap** 不显著 —— 不是独立的解释因子
- ⚠️ **Absolute rarity** 严重力量不足（仅 5 对/28 pairs after expansion），Func-TCD 趋近边缘显著 (p=0.084) 但无法定论

### 5.6 RT：真实文本生态效度验证

**结果**（`results/real_text_validation/real_text_report.txt`）：

**设计**：12 篇自然文本段落（新闻、专业传记、法律证词、医疗记录、教育报告、推荐信、社区描述 7 个类别）× 10 查询 × 14 名字对 = **1,680 次测试**

| 设置 | Func-TCD | Cont-TCD | Ratio | p |
|------|----------|----------|-------|---|
| Synthetic templates | 0.0177 | 0.0133 | 1.44× | 2.2e-47 |
| **Natural text** | **0.0124** | **0.0087** | **1.42×** | **6.4e-26** |
| BM25 baseline | 0.0000 | 0.0000 | — | — |

**关键发现**：
- 功能词优势比从 **1.44× 降至 1.42×** —— 几乎完全一致！
- 在所有 7 个文本类别中模式一致（func > cont 胜率在 9/12 段落中 > 50%）
- 真实文本的绝对 TCD 值较低（0.0124 vs 0.0177），因为自然文本的语言多样性稀释了信号，但**比率**保持不变

### 5.7 BM25：非上下文模型基线对照

**结果**（`results/bm25_baseline/bm25_report.txt`）：

| 指标 | ColBERT | BM25 |
|------|---------|------|
| Mean Func-TCD | 0.012371 | **0.000000** |
| Mean Cont-TCD | 0.008711 | **0.000000** |
| Ratio | 1.42× | **0.00×** |

**BM25 产生了精确的零功能词 TCD**。这是预期的：BM25 基于精确词频匹配（TF-IDF），更换名字只影响名字 Token 本身的得分，不会"泄漏"到其他 Token。

**结论**：功能词偏见传导是**上下文表示模型 (contextual representations)** 的特有现象，不是检索分解本身的副产物。

---

## 6. 核心发现汇总与可引用结论

### 6.1 安全可引用的结论

> **关于功能词传导**："ColBERT 中的身份偏见不是通过语义内容词（如 'doctor'、'nurse'）传导的，而是通过功能词（如 'is'、'who'、'a'）传导。功能词的 TCD 是内容词的 **1.44×**，在 33 个职业中有 30 个成立，在自然文本中保持 1.42× 的几乎相同比率，且在 BM25 中完全不存在。"

> **关于罕见度放大**："BPE tokenization 是一个隐性的公平性税：少数族裔名字更有可能被切碎成多个 subword token，从而放大偏见 1.84×。"

> **关于混淆因素**："经典的名字替换审计测量的不是单一种族效应。在 43,065 次控制实验中，最稳定的信号来自 cross-gender 和 tokenization mismatch。cross-race 在完整模型中不稳定，仅在靶向验证中显示较弱的残留信号。"

> **关于身份特异性**："功能词的偏见放大不是 ColBERT 对所有词汇替换的通用敏感性 —— 身份替换的 Func-TCD 显著高于非身份替换 (p < 1e-8, d = 0.37)。"

### 6.2 不安全、不应声称的结论

- ❌ 不要说 "P2 证明了稳定独立的 race main effect"（它在完整规格中减弱到不显著）
- ❌ 不要说 "absolute rarity 已被干净识别"（仅 5 对，力量不足）
- ❌ 不要说 "频率差距不重要"（它在完整样本中显著，只在严格匹配子集中减弱）
- ❌ 不要说 Rosenman 数据可以代表全美人口（它基于 6 个南方州的选民档案）

---

## 7. 代码架构与文件索引

### 7.1 目录结构

```
colbert-bias-audit/
├── src/                                    # 核心算法封装
│   ├── audit/
│   │   └── core.py                         # ★ 126行：encode, maxsim_detail, classify_token,
│   │                                       #   compute_tcd_breakdown — 所有实验的基础
│   ├── models/                             # 模型 wrapper（当前仅 __init__.py）
│   ├── metrics/                            # 指标定义（当前仅 __init__.py）
│   └── visualization/                      # 可视化工具（当前仅 __init__.py）
│
├── data/
│   ├── audit_names/
│   │   ├── name_features.json              # 初始 30 名字特征矩阵
│   │   ├── name_features_tokenizer_verified.json  # tokenizer 复核版本
│   │   ├── p2b_rosenman_name_features.json # Rosenman 扩展后 65 名字特征
│   │   ├── p2b_rosenman_matrix.json        # ★ Rosenman 5-family 匹配对矩阵（253KB）
│   │   ├── p2b_rosenman_additions.json     # 手动性别标注的新增名字
│   │   ├── p2b_verified_matrix.json        # 旧版 verified 矩阵（已被 Rosenman 版取代）
│   │   ├── source_registry.json            # 外部数据源注册表
│   │   └── external/                       # Rosenman 原始数据表
│   │       ├── first_nameRaceProbs.tab
│   │       └── first_raceNameProbs.tab
│   ├── professions/
│   │   └── professions.json                # 33 职业元数据（query, BLS category, female%）
│   └── counterfactual/                     # （预留目录）
│
├── experiments/
│   ├── phase0_control/
│   │   └── p0_control_experiment.py        # ★ 1,080 组身份特异性对照
│   ├── phase1_synthetic/
│   │   ├── targeted_experiments.py         # ★ 783 对：代词偏见 + 职业特异性
│   │   ├── function_word_bias.py           # ★ 650 对：功能词发现 + 嵌入扰动分析
│   │   ├── token_level_audit.py            # 12 对：初始 Token 级审计
│   │   └── rarity_profession_audit.py      # ★ 1,140 组：罕见度中介效应
│   ├── phase2_profession_expansion/
│   │   ├── profession_expansion.py         # P1 33 职业扩展
│   │   └── p1_wrapup_analysis.py           # P1 收尾统计分析
│   ├── phase2_name_confound/               # ★ 当前深水区
│   │   ├── name_confound_decomposition.py  # ★ P2 主回归（43,065 测试）
│   │   ├── p2_final_analysis.py            # P2 最终分析 + 模型阶梯
│   │   ├── p2_followup_analysis.py         # P2 补充分析
│   │   ├── p2_orthogonalized_panel.py      # 正交子集面板
│   │   ├── build_p2b_rosenman_matrix.py    # ★ Rosenman 矩阵构建
│   │   ├── p2b_rosenman_validation.py      # ★ Rosenman 靶向回归验证
│   │   ├── build_p2b_verified_matrix.py    # 旧版 verified 矩阵构建
│   │   ├── p2b_verified_matrix_analysis.py # 旧版分析
│   │   ├── p2b_aux_names_dataset.py        # names-dataset 辅助验证（已降级）
│   │   ├── p2b_gap_report.py               # 匹配对 gap 报告
│   │   └── scan_rarity_candidates.py       # 罕见度候选名字扫描
│   ├── phase2_real_text/                   # （预留目录）
│   └── phase3_multi_model/                 # （预留目录）
│
├── results/
│   ├── finding_function_word_bias.md       # ★ 发现 A 正式文档
│   ├── finding_rarity_effect.md            # ★ 发现 B 正式文档
│   ├── p1_profession_expansion/            # P1 结果（11 文件含 CSV、PNG、JSON）
│   ├── p2_name_confound/                   # ★ P2/P2b 结果（34 文件，含 14.9MB 原始 CSV）
│   ├── real_text_validation/               # RT 结果（4 文件）
│   └── bm25_baseline/                      # BM25 结果（4 文件）
│
├── paper/
│   ├── main.tex                            # ★ ACM FAccT 格式论文主稿（411 行）
│   ├── refs.bib                            # 参考文献
│   ├── generate_figures.py                 # 论文图表生成脚本
│   ├── figures/                            # 7 组论文图表（PDF + PNG）
│   ├── main.md                             # Markdown 版论文草稿
│   ├── p2_lit_review_progress_update.md    # P2 相关文献综述
│   ├── overall_lit_review_progress_update.tex  # 完整文献综述
│   └── overall_lit_review_refs.bib
│
├── PROJECT_INTRODUCTION.md                 # ← 本文档
├── README.md                               # 简版项目说明
├── research_roadmap.md                     # 研究路线图
├── AGENT_CONTEXT_P2_HANDOFF.md            # P2 Agent 交接文档（面向 AI）
├── project_overview.html                   # 中文项目讲解（可视化页面）
├── formal_findings.html                    # 正式研究发现总结页面
└── progress_tracker.html                   # 进度看板页面
```

### 7.2 执行命令速查

```bash
# 环境准备
conda create -n colbert_bias python=3.10
conda activate colbert_bias
pip install torch transformers matplotlib seaborn scipy statsmodels pandas

# P0 对照实验
conda run -n colbert_bias python experiments/phase0_control/p0_control_experiment.py

# Phase 1 功能词发现
conda run -n colbert_bias python experiments/phase1_synthetic/function_word_bias.py

# Phase 1 罕见度效应
conda run -n colbert_bias python experiments/phase1_synthetic/rarity_profession_audit.py

# P1 职业扩展
conda run -n colbert_bias python experiments/phase2_profession_expansion/profession_expansion.py

# P2 主回归
conda run -n colbert_bias python experiments/phase2_name_confound/name_confound_decomposition.py

# P2b Rosenman 矩阵构建
conda run -n colbert_bias python experiments/phase2_name_confound/build_p2b_rosenman_matrix.py

# P2b Rosenman 靶向验证
conda run -n colbert_bias python experiments/phase2_name_confound/p2b_rosenman_validation.py

# 论文图表生成
conda run -n colbert_bias python paper/generate_figures.py
```

---

## 8. 论文状态与投稿目标

### 8.1 论文手稿

当前论文 (`paper/main.tex`) 为 ACM FAccT 格式（`\documentclass[sigconf,review,anonymous]{acmart}`），411 行，包含：
- Abstract（完整）
- Introduction（完整，5 个贡献点）
- Related Work（完整，4 段：fairness in retrieval、names as identity proxies、tokenization and representation、ColBERT and late interaction）
- Methodology（完整，含 TCD 数学定义、名字池设计、5 个实验阶段描述）
- Results（完整 4 节：function-word bias、rarity amplification、confound decomposition、ecological validation + BM25）
- Discussion（完整 4 节：distributed bias、tokenization as infrastructure-level bias、ColBERT as diagnostic、debiasing implications）
- Limitations（5 项）
- Conclusion（完整）

### 8.2 论文图表

`paper/figures/` 包含 7 组图表（PDF 用于 LaTeX 嵌入，PNG 用于预览）：
1. `tcd_schematic` —— TCD 计算流程示意图
2. `func_vs_cont_professions` —— 33 职业的 Func-TCD vs Cont-TCD
3. `name_vs_pronoun` —— Name swap vs Pronoun swap 效应对比
4. `rarity_amplification` —— 罕见度放大效应
5. `decomposition_forest` —— P2b 系数森林图
6. `model_ladder` —— 嵌套模型 R² 阶梯
7. `cross_setting_validation` —— 跨设置验证（Template vs Natural text vs BM25）

### 8.3 投稿目标

| 会议 | 截稿日期 | 匹配度 |
|------|---------|--------|
| **FAccT 2026** | 已提交格式 | ⭐⭐⭐⭐⭐ 公平性专场 |
| **EMNLP 2026** | ~6 月 | ⭐⭐⭐⭐⭐ NLP + Fairness |
| **NAACL 2026** | ~10 月 | ⭐⭐⭐⭐ NLP |
| **CIKM 2026** | ~5 月 | ⭐⭐⭐ IR + Data Mining |

---

## 9. 已知局限与未解决问题

| # | 问题 | 严重性 | 当前状态 |
|---|------|--------|---------|
| 1 | **Absolute rarity 力量不足** | 🔴 高 | 仅 5 对（28 组合对），p=0.084 边缘显著但无法定论 |
| 2 | 单模型（仅 ColBERTv2） | 🟡 中 | BM25 对照已完成；DPR/SPLADE 对比待做 |
| 3 | Rosenman 数据地域偏差 | 🟡 中 | 基于 6 个南方州选民档案，需在论文中披露 |
| 4 | 名字池仅覆盖美国 4 类种族 | 🟡 中 | White, Black, Hispanic, Asian；不含其他地域名字 |
| 5 | 无去偏评估 | 🟢 低 | Null-Space Projection 为 stretch goal，论文已定位为测量/诊断型 |
| 6 | 主要使用合成/半合成数据 | 🟢 低 | 真实文本验证已完成（1,680 测试），比率一致 |

---

## 10. 下一步行动清单

### 优先级 1（当务之急）：补完 P2b absolute_rarity

**目标**：将 absolute_rarity 的对数从 5 对提升至 15-20 对，使其具有统计力量。

**方法**：在 Rosenman 数据集中定向搜索满足以下条件的新名字：
- 与已有名字相同 race、相同 gender、相同 BPE token count
- 与配对名字相近的 frequency
- 但 mean log-frequency 有明显差异（一低一高）

**入口文件**：`data/audit_names/p2b_rosenman_additions.json`
**构建脚本**：`experiments/phase2_name_confound/build_p2b_rosenman_matrix.py`
**验证脚本**：`experiments/phase2_name_confound/p2b_rosenman_validation.py`

### 优先级 2：论文打磨

- 确保所有数字与最新 P2b 结果一致
- 完成 limitation 部分的 absolute rarity 更新
- 检查所有图表是否反映最新数据

### 优先级 3（如有余力）：去偏实验

- 提取功能词跨反事实对的 embedding 差异向量 δ
- PCA/SVD 提取身份敏感方向
- Null-Space Projection 到零空间
- 测量去偏后 SS 下降幅度 vs 检索质量 (MRR@10, NDCG@10) 下降幅度

---

## 11. AI Agent 决策指南

### 11.1 不可变原则

1. **不要引入 Rosenman 以外的第三个新名字数据库**。当前方法论已收敛，增加数据源只会增加混乱。
2. **不要试图证明"稳定独立的 race main effect"**。当前数据不支持这个结论，强行声称只会被审稿人打回。
3. **保持保守的科研表述**。所有对外声称必须有明确统计支撑。
4. **优先改进证据质量（扩充 absolute_rarity）而非规模（跑更多职业或模板）**。

### 11.2 关键文件阅读优先级

如果你是新接手这个项目的 Agent，按以下顺序阅读：

1. `src/audit/core.py` —— 理解所有度量的实现
2. `results/p2_name_confound/p2b_rosenman_report.txt` —— 最新量化结果
3. `results/p2_name_confound/p2_final_report.txt` —— 嵌套模型阶梯
4. `results/real_text_validation/real_text_report.txt` —— 生态效度
5. `results/bm25_baseline/bm25_report.txt` —— BM25 对照
6. `AGENT_CONTEXT_P2_HANDOFF.md` —— P2 详细交接文档
7. `paper/main.tex` —— 论文当前状态

### 11.3 环境信息

- Python 3.10, conda 环境名 `colbert_bias`
- 核心依赖：`torch`, `transformers`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`, `pandas`
- 模型：`colbert-ir/colbertv2.0`（HuggingFace）
- 自动设备选择：CUDA > MPS > CPU
- 学校 HPC：Brown University Oscar 集群

---

## 12. EMNLP 模拟审稿报告

> **模拟审稿人背景**：NLP fairness + dense retrieval 方向，熟悉 EMNLP/ACL/ARR 审稿标准
>
> **审稿日期**：2026-04-20
>
> **论文**：*Where Does Bias Hide? Token-Level Attribution Reveals Distributed Bias Propagation in Late-Interaction Retrieval*

### 12.1 总体评分

| 评审维度 | 分数 (1-5) | 说明 |
|---------|-----------|------|
| **Soundness（技术正确性）** | 3.5 | 核心方法正确，但合成数据 + 单模型构成实证瓶颈 |
| **Novelty（原创性）** | 4.0 | TCD 作为 token 级偏见归因指标是新颖的；功能词传导发现具有意外性 |
| **Significance（重要性）** | 3.5 | 对理解检索偏见机制有价值，但缺少下游任务影响量化 |
| **Clarity（写作清晰度）** | 4.0 | 结构清晰，数学定义精确，图表质量高 |
| **Reproducibility（可复现性）** | 3.0 | 代码/数据声明不足，HuggingFace 模型可用但名字矩阵未说明是否开源 |
| **Overall（总评）** | 3.5 | Borderline accept — 需要修改后具有 EMNLP 竞争力 |

**推荐决定：Revise（修改后重审）**

---

### 12.2 摘要评价

本文利用 ColBERT（Late Interaction 架构）的 MaxSim 可分解性提出了 TCD（Token Contribution Disparity）指标，揭示了一个违反直觉的发现：身份偏见在检索模型中主要通过功能词（`is`, `who`, `a`）而非内容词（`doctor`, `nurse`）传导。实验覆盖 43,065 次反事实测试、33 个职业、65 个名字，并通过 BM25 对照和真实文本验证建立了泛化性。

本文的方法论贡献（TCD）和核心实证发现（分布式偏见传导）是扎实的。然而，EMNLP 审稿人关注的是**实证完备性**：论文在 benchmark 评估、多模型比较、下游任务影响三个方面存在明显不足，且部分统计分析（absolute rarity, embedding perturbation）力量不够。

---

### 12.3 优点（Strengths）

**S1. 方法论贡献清晰且有意义**

TCD 是目前第一个针对密集检索模型的 token 级偏见归因指标。之前的 retrieval fairness 工作（Rekabsaz et al. 2021, Ziems et al. 2024）停留在 aggregate ranking level，无法回答"偏见从哪个 token 来"。ColBERT 的 MaxSim 天然支持这种分解，作者准确抓住了这个 architectural affordance。TCD 的数学定义简洁（Eq. 2），物理意义明确，且可以推广到任何保留 per-token score 的检索架构。

**S2. 核心发现具有意外性（Surprising Finding）**

"偏见不是来自 'doctor' 或 'nurse'，而是来自 'is'"——这一发现直接挑战了 NLP 偏见研究中最常见的叙事（lexical association bias，如 Bolukbasi et al. 2016 的 "man is to computer programmer"）。这种 counterintuitive finding 对 EMNLP 社区有很高的叙事吸引力，也具备实际意义：如果偏见不在内容词里，那么传统的 word-level debiasing（如 gender-direction removal）从根本上就瞄错了靶子。

**S3. 实验控制设计严谨**

- **P0 Control**：将身份替换 vs 非身份替换（如 "ten → fifteen"）作为对照，证明效应的身份特异性（p < 1e-8, d=0.37），排除了"模型对任何替换都敏感"的 null hypothesis。
- **BM25 Baseline**：产生精确的零功能词 TCD，干净地证明效应是上下文表示的产物，不是检索分解框架的 artifact。
- **Matched-pair family design**：P2b 的 5 族匹配对设计（固定 3 个变量、仅变 1 个）在方法论上接近准实验设计。

**S4. 嵌套模型阶梯（Model Ladder）分析**

逐步添加控制变量并观察 cross_race 从显著→不显著的衰减轨迹，展示了坦诚的分析态度。许多 bias 论文会过度声称 race effect；本文反而在完整规格中承认 race 不稳定，这增加了可信度。

**S5. 跨设置一致性**

Synthetic 1.44× → Natural text 1.42× 的比率一致性是令人信服的。这种定量的跨域稳定性在 bias 研究中罕见，显著提升了文章的 generalizability argument。

---

### 12.4 弱点（Weaknesses）

**W1. 🔴 缺少标准 IR Benchmark 评估（Critical）**

这是论文在 EMNLP 最大的实证缺口。EMNLP 社区对 retrieval 论文的期望是在 **MS MARCO、TREC DL、BEIR** 等标准 benchmark 上进行评估。当前论文的所有实验都基于：

- (a) 合成模板（`{name} has extensive experience in this field.`）
- (b) 研究者手写的 12 篇"自然文本"段落

虽然作者在 Limitations §6 中承认了这一点，但 EMNLP 审稿人通常不会接受"我们知道应该做但没做"作为充分的解释。

**具体缺失**：
- 未在 MS MARCO passages 中进行 NER-based name extraction + counterfactual swap
- 未报告 TCD 在真实检索排名（MRR@10, NDCG@10, Recall@100）上的影响
- 所有 "natural text" 仍然是 researcher-constructed，不是从语料库中自然采样的

> **修改建议**：从 MS MARCO dev set 中抽样 500-1000 个包含人名的 passage，使用 NER 识别名字，构建 counterfactual pairs，然后：
> (1) 计算 TCD 并验证功能词传导模式是否保持
> (2) 测量 identity swap 对实际检索排名（MRR, NDCG）的影响
> (3) 这将同时解决"生态效度"和"下游影响"两个问题

**W2. 🔴 仅测试一个 Contextual Retriever（Critical）**

论文标题使用了 "Late-Interaction Retrieval"，暗示发现是 ColBERT specific 的。但 Introduction 和 Discussion 中多次将结论泛化到 "transformer-based retrieval"、"contextual representations"——这需要多模型证据支撑，而当前只有 ColBERTv2。

EMNLP 审稿人会合理要求至少两种对比：

| 模型 | 可做的分析 | 理由 |
|------|-----------|------|
| **SPLADE** | TCD-analogous 分析（SPLADE 有 per-token expansion weights） | 验证 "token-level bias propagation" 是否在 learned sparse 模型中也存在 |
| **DPR** | Aggregate-level SS（总分敏感性）对比 | 即使 DPR 不能做 token-level 分解，aggregate bias 对比能说明 ColBERT 是在"诊断共同疾病"还是"只有自己有病" |
| **Cross-encoder (e.g., monoT5)** | Aggregate-level SS | 验证全交互模型 vs Late Interaction 的 bias 差异 |

> **修改建议**：至少添加 DPR aggregate-level 对比（实现成本很低，仅需 cosine similarity + counterfactual pairs）+ SPLADE per-term weight 分析（如可行）。如果只能做一个，优先 DPR。

**W3. 🟡 Embedding Perturbation 分析力量不足**

§4.1 声称 "function-word embeddings shift 8.9× more than content words (p = 0.0002)"。但细查实验代码（`function_word_bias.py` L324-358）发现这个分析：

- 仅基于 **3 个测试用例**（`He→They`, `Emily→Lakisha`, `Greg→Jamal`）
- 涉及 9 个功能词观测 + 6 个内容词观测（**总共 15 个数据点**）
- 8.9× 和 p=0.0002 来自如此小的样本，可能不稳定

论文中报告时给人的印象是大规模验证的结论，但实际是小样本结果。

> **修改建议**：将 embedding perturbation 分析扩展到所有 650 个反事实对（或至少一个有意义的子集，如 100 对），然后重新计算 shift ratio 和显著性。如果不扩展，需在论文中明确标注 N=15 和"illustrative"性质。

**W4. 🟡 功能词 vs 内容词的分类方式缺乏敏感性分析**

功能词列表（`src/audit/core.py` L15-22）是手动定义的，包含 56 个词。核心发现（1.44× ratio）对这个列表的定义高度敏感：

- 如果将某些 "边界词"（如 `has`, `been`）重新分类，ratio 是否会显著变化？
- 如果使用 POS-tagger 自动分类（determiner/auxiliary vs noun/verb/adjective），结论是否一致？
- 特殊 token（`[CLS]`, `[SEP]`, `?`）被分别处理了吗？（代码显示 `?` 被归入功能词，这可能有争议）

> **修改建议**：添加一个 ablation study（在 Appendix 也可）：
> (1) Remove punctuation (`?`, `.`, `,`) from function words → recalculate ratio
> (2) Use spaCy POS tags for automatic classification → compare
> (3) Report sensitivity of the 1.44× ratio to ±3 changes in the function word list

**W5. 🟡 方向性分析（Directionality）不在正文中**

论文证明了功能词 TCD **magnitude** 更大，但在正文中完全没有报告 **direction**：功能词的扰动是否系统性地偏向某个群体？

- 代码数据（`function_word_bias.py` Analysis 2）显示 `is` 72.3% 偏向 A（通常是多数群体名字），`?` 68.6%，`a` 64.3%。这些数据极有价值但仅存在于实验输出，从未进入论文。
- 对于 EMNLP 审稿人来说，仅知道"功能词扰动更大"是不够的——还需要知道"扰动朝同一个方向"（systematic bias vs random noise）。

> **修改建议**：在 §4.1 添加一个方向性段落，报告：
> - 功能词 TCD 偏向 majority-group name 的比例（across all pairs）
> - 与 50% 基线的二项检验
> - 这将把 finding 从 "function words are more volatile" 提升为 "function words systematically favor dominant-group identities"

**W6. 🟡 3% Score Sensitivity 的实际检索影响未知**

SS ≈ 3% 意味着两个反事实文档的分数差异约为总分的 3%。论文从未回答：**这 3% 会不会在实际检索中改变排名？**

- 如果一个 candidate pool 中有 1000 篇文档，3% 的分数差可能完全被其他文档的 score gap 淹没
- 如果只有 2-3 篇高度相关的文档，3% 可能翻转 top-1

不需要 full-scale retrieval evaluation 来回答这个问题——可以用 synthetic retrieval experiment（构建 mini-corpus，测量 rank changes）。

> **修改建议**：构建一个 synthetic retrieval scenario（e.g., 10 documents of varying relevance + 2 counterfactual identity documents），测量 identity swap 导致 rank position 变化的频率和幅度。

**W7. 🟡 Absolute Rarity 不应保留在主表中**

Table 4 中 absolute rarity 的 N=28, p=0.084 在 EMNLP 审稿中会被标记为 underpowered：

- 严格的 EMNLP 审稿人可能认为"margin of significance" framing 接近 p-hacking
- 在一个与 4 个显著/不显著因素并列的表格中，它的存在混淆了主叙事

> **修改建议**：要么 (a) 从 Table 4 移除，转入"Exploratory Analysis"段落并加注 underpowered caveat；要么 (b) 在 resubmission 前扩充到 ≥15 对并获得干净的显著/不显著结论。

**W8. 🟢 Related Work 缺少 2024-2025 重要对比**

- **TokenShapley (2025)**：基于 Shapley value 的 token-level attribution，概念上与 TCD 有重叠但方法论不同。需讨论两者差异：TCD 是 counterfactual-based per-token attribution，TokenShapley 是 data-attribution-based。
- **KLAAD (EMNLP 2025)**：attention-based debiasing 框架，将 attention 分布在 stereotypical vs anti-stereotypical 句子之间对齐。与本文的 "bias concentrates in function words" 发现形成有趣对比——KLAAD 的方法隐含假设 bias 可以通过 attention redistribution 解决，而本文发现 bias 在 embedding space（非 attention weights）中传播。
- **EqualizeIR (2025)**：检索公平性的 regularization 方法。应讨论 TCD 如何可能作为 EqualizeIR 的诊断补充工具。
- **Manchanda & Shivaswamy (2025)** 虽已引用，但讨论太简略。他们的 anonymization 方法正是本文 finding 的一个潜在应用场景。

**W9. 🟢 可复现性声明不足**

EMNLP 从 2023 年起强烈鼓励提交 Reproducibility Checklist。当前论文缺少：

- 计算资源说明（GPU 型号、运行时间）
- 代码/数据开源声明（是否会释放名字矩阵、实验脚本？）
- 随机种子说明（代码中 `np.random.seed(42)` 存在，但论文未提及）
- 超参数说明（无 hyperparameters 需要调，但应明确声明 TCD 是 hyperparameter-free）

**W10. 🟢 论文格式需从 ACM sigconf 转为 ARR/EMNLP 格式**

当前 `\documentclass[sigconf,review,anonymous]{acmart}` 是 ACM 格式。EMNLP 通过 ARR 投稿，需要使用 `acl_latex` 模板（https://github.com/acl-org/acl-style-files）。这不仅是格式问题——页数限制、参考文献格式、Limitations section 的位置都不同。

---

### 12.5 EMNLP 审稿人会问的关键问题

**Q1.** "在 MS MARCO 或任何标准 IR benchmark 上，identity swap 会导致多大的 MRR/NDCG 下降？你的 3% SS 在真实检索场景中有 practical significance 吗？"

**Q2.** "DPR 或 SPLADE 是否也展现出类似的分布式偏见传导模式？如果 DPR 的 aggregate bias 更大但不可诊断，这说明什么？如果 SPLADE 的 learned expansion tokens 也 carry bias，这对你的 'contextual representation phenomenon' 论述有什么影响？"

**Q3.** "功能词分类列表是手动定义的。你测试过用 POS tagger 自动分类后结论是否一致吗？将 '?' 从功能词列表中移除后 ratio 是否显著变化？"

**Q4.** "Table 1 的 N 列为 '---'。请提供确切的样本量。"

**Q5.** "你的 3 个模板结构高度相似（`{name} has/is/has been...`）。你是否测试过句法多样性更高的模板（被动语态、关系从句、倒装句）？模板结构是否可能混淆了 token position effects 和 function word effects？"

**Q6.** "Blodgett et al. (2020) 和 Jain & Wallace (2019) 出现在 refs.bib 中但从未在正文中引用。这是有意为之还是遗漏？如果引用了 Blodgett et al.，你如何回应其关于 'who is harmed' 的核心批评？"

---

### 12.6 与 FAccT Review 的关键差异

| 维度 | FAccT 审稿人关注 | EMNLP 审稿人关注 |
|------|----------------|----------------|
| **Harm model** | 🔴 Critical — 必须说明"对谁造成了什么伤害" | 🟡 Important — 需要 practical significance 论证，但不要求 harm taxonomy |
| **Normative framing** | 🔴 Critical — 需要引用 Blodgett (2020) 并正面回应 | 🟢 Nice-to-have — 只需合理的 motivation |
| **Intersectionality** | 🟡 Expected — 需要 race×gender 交叉分析 | 🟢 Optional — 如果有更好，没有不扣分 |
| **Benchmark evaluation** | 🟢 Nice-to-have | 🔴 **Critical** — MS MARCO / BEIR 几乎是必须的 |
| **Multi-model comparison** | 🟡 Important | 🔴 **Critical** — 至少需要 DPR 对比 |
| **Ablation studies** | 🟡 Important | 🔴 **Critical** — 功能词列表 sensitivity, template diversity |
| **Reproducibility** | 🟢 not required | 🟡 **Important** — checklist, code release statement |
| **Positionality statement** | 🟡 Expected (camera-ready) | 🟢 Not expected |
| **Downstream task impact** | 🟡 Important | 🔴 **Critical** — SS → ranking change 量化 |
| **Ethics statement** | 🟡 Important | 🟡 Important (Limitations section is mandatory) |

---

### 12.7 EMNLP 投稿前优先修改清单

#### Tier 1：必须完成（否则大概率被拒）

| # | 修改项 | 工作量 | 影响 |
|---|--------|--------|------|
| E1 | **MS MARCO counterfactual evaluation**：从 dev set 提取含人名 passage → NER → swap → 测 TCD + MRR/NDCG 变化 | 高（~2周） | 🔴 解决 W1（benchmark）+ W6（ranking impact）两个 critical 弱点 |
| E2 | **添加 DPR aggregate-level 对比**：对相同 counterfactual pairs 计算 DPR cosine similarity SS | 中（~3天） | 🔴 解决 W2（single model）|
| E3 | **方向性分析进正文**：报告功能词 TCD systematic direction（% favoring majority group, binomial test） | 低（~1天） | 🔴 把 finding 从 "volatile" 提升为 "systematically biased" |
| E4 | **论文格式转为 ARR/ACL 模板** | 低（~1天） | 🔴 格式不对直接 desk reject |

#### Tier 2：强烈建议（显著提升接受概率）

| # | 修改项 | 工作量 | 影响 |
|---|--------|--------|------|
| E5 | **功能词列表 sensitivity ablation**：POS-tagger 版本 + 去标点版本 + leave-k-out | 中（~3天） | 🟡 解决 W4 |
| E6 | **扩展 embedding perturbation 到 ≥100 对** | 中（~3天） | 🟡 解决 W3 |
| E7 | **Absolute rarity 扩充或降级** | 中（~3天） | 🟡 解决 W7 |
| E8 | **添加 SPLADE per-term weight 分析**（如条件允许） | 高（~1周） | 🟡 显著增强多模型论证 |
| E9 | **Reproducibility checklist + code release statement** | 低（~半天） | 🟡 解决 W9 |

#### Tier 3：加分项

| # | 修改项 | 工作量 | 影响 |
|---|--------|--------|------|
| E10 | 补充 TokenShapley, KLAAD, EqualizeIR 到 Related Work | 低 | 🟢 |
| E11 | 模板多样性 ablation（被动语态、关系从句） | 中 | 🟢 |
| E12 | 修复 Table 1 的 N=--- 和未引用的 bib entries | 极低 | 🟢 |
| E13 | 添加非二元代词结果（代码已有数据） | 低 | 🟢 |

---

### 12.8 总体判断

**作为 EMNLP 审稿人，我给出 Borderline Accept (3.5/5) 的评分，倾向于 Revise。**

**核心判断逻辑**：

- **接受理由**：TCD 方法论在 token-level bias attribution 领域是 genuinely novel 的；功能词传导发现具有 surprising and counterintuitive 的特质；实验控制（P0, BM25 baseline）设计优秀；confound decomposition 方法论严谨
- **拒绝风险**：无标准 benchmark 评估（EMNLP IR track 几乎不会接受没有 MS MARCO/BEIR 的论文）；单模型限制泛化声称；3% SS 的 practical significance 不明；部分分析力量不足

**如果作者完成 E1（MS MARCO）+ E2（DPR）+ E3（方向性）+ E4（格式），论文竞争力会从 borderline 提升到 solid accept 区间。**

这篇论文的故事 ("bias hides in 'is', not in 'doctor'") 有极强的叙事吸引力，适合 EMNLP 的 Fairness & Interpretability track。关键是用标准 benchmark 的实证证据来支撑这个故事。

---

*文档最后更新：2026-04-20*
*项目状态：P2b Rosenman 靶向验证完成，absolute_rarity 待扩充，论文主稿基本完成*
*附录：EMNLP 模拟审稿报告已附在本文档末尾*
