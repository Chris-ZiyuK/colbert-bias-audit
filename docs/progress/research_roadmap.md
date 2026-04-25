# Research Roadmap: Token-Level Bias Attribution in Dense Retrieval

**Chris Ziyu Kong · CSCI 2952W · Last updated: March 9, 2026**

---

## Part I · 研究脉络：从模糊的想法到精确的发现

### 起点：一个模糊的"如果呢"

最初的想法非常简单：**搜索引擎会不会因为你的名字而歧视你？** 

这个想法来自两个源头：
- 课堂上 Bowker & Star (1999) 关于"分类的政治学"的讨论 — 分类系统从来不是中性的；
- 劳动经济学中经典的 **Bertrand & Mullainathan (2004)** 简历审计实验 — "Emily" 和 "Lakisha" 投同一份简历，前者的回调率高 50%。

我把这个逻辑平移到了信息检索领域：如果把简历里的名字换掉，ColBERT 这种搜索模型给出的相关性分数会不会不一样？

### 第一阶段：验证偏见"有没有"（RQ1）

**实验：** `quick_experiment.py` → `expanded_experiment.py` → `targeted_experiments.py`

我从 8 对反事实文档开始，逐步扩大到 **783 对**，覆盖了种族（White vs Black vs Hispanic vs Asian）和性别（He / She / They）两个维度，横跨多种职业。

**核心发现：**
- ✅ **偏见确实存在**。ColBERTv2 在替换名字/代词后，给出了统计显著的不同分数（p < 0.001）。
- ✅ **非二元性别惩罚**。使用 "They" 代词的文档被系统性地降分，而 He/She 之间的差异不显著。
- ✅ **职业特异性**。"talented researcher" 和 "reliable nurse" 的偏见幅度远大于 "experienced doctor"。

**意义**：这一步本身不是原创的 — 之前有人做过类似的 Dense IR 偏见检测（如 Rekabsaz et al., 2021）。但它验证了方向的可行性，为下一步的创新提供了数据基础。

---

### 第二阶段：定位偏见"在哪里"（RQ2）— 我们的核心创新

#### 转折点：利用 ColBERT 的"后期交互"做白盒审计

ColBERT 与 DPR 等模型的根本区别在于：它保留了每个词的独立向量，并通过 **MaxSim**（逐词最大相似度求和）来计算总分。这意味着总分可以被**拆解**为每个查询词的贡献 — 我们发明了一个指标叫 **TCD（Token Contribution Disparity）**，来量化每个查询词在身份替换后的分数变化。

**实验：** `token_level_audit.py`（12 对） → `function_word_bias.py`（650 对）

### 🔬 发现 A：分布式偏见传导（Function-Word Bias Leakage）

> **偏见不是通过 "doctor" 或 "nurse" 这些语义词传导的，而是通过 "is"、"who"、"a" 这些功能词间接传导。**

| 指标 | 功能词 | 内容词 | 比率 | 显著性 |
|------|--------|--------|------|--------|
| Mean \|TCD\| | 0.0210 | 0.0146 | **1.44×** | p = 0.0002 *** |
| Embedding Shift | 0.1137 | 0.0128 | **8.9×** | p = 0.0002 *** |

**在 10 个职业中，9 个都呈现出功能词 > 内容词的 TCD 模式。**

**为什么会这样？** BERT 的 Self-Attention 会让身份替换的影响传播到句子里的每一个词。内容词（如 "clinical"）有强语义锚点，向量很稳定；功能词（如 "is"）几乎没有独立语义，是纯粹的"上下文海绵"，因此吸收了最多的身份扰动。ColBERT 的 MaxSim 机制进一步放大了这种效应。

**创新意义**：这推翻了"偏见 = 词汇关联（nurse ↔ female）"的传统认知。偏见在上下文化模型中是**分布式的、弥漫的**，传统去偏方法（修某个词的向量）可能完全无效。

---

#### 🔬 发现 B：罕见度中介效应（Rarity-Mediated Identity Effect）

> **名字越罕见（被 BPE 切分的 Token 越多），功能词吸收的偏见越多，总分偏差越大。**

**实验：** `rarity_profession_audit.py`（1,140 组测试）

| 名字组合 | Mean SS（总偏见） | Mean Func-TCD | 倍率 |
|----------|:---:|:---:|:---:|
| Common-Common | 0.0210 | 0.0108 | 1.0× |
| Common-Rare | 0.0305 | 0.0163 | 1.5× |
| **Rare-Rare** | **0.0366** | **0.0199** | **1.84×** |

**ANOVA 结果**：Profession（F=214, p<0.0001）和 Rarity（F=300, p<0.0001）都是极其显著的偏见预测因子。

**为什么会这样？** 罕见名字（如 "Thiruvengadam"）被切成多个 subword token，增加了模型的不确定性。模型在"拼凑"陌生身份时，更依赖训练数据中的刻板印象来"填空"，导致偏见放大。**罕见度是身份的"隐形税"。**

**创新意义**：这揭示了 BPE 分词器本身就是偏见的放大器 — 少数族裔的名字更有可能被切碎，从而承受更大的偏见惩罚。这不仅是一个检索问题，也是一个**基础设施公平性**问题。

---

### 我们现在站在哪里？

```
Phase 1 ✅              Phase 2 ✅               Phase 3 (Now)          Phase 4
─────────────        ──────────────────       ─────────────────      ──────────────
偏见检测 (783对)       Token级归因 (1,790对)      真实语料 + 多模型        去偏干预
SS, p-value          TCD, Heatmap, Gini       MS MARCO / BEIR        Null-space Proj
合成模板              功能词发现 + 罕见度效应     生态效度验证             Utility-Fairness
```

---

## Part II · 下一阶段实验计划

### 目标：两个月内投稿顶会（EMNLP 2026 / FAccT 2026 / SIGIR 2026）

---

### Phase 3：生态效度验证（预计 3 周）

**核心问题**：功能词偏见传导和罕见度效应是合成模板的产物，还是在真实文本中也存在？

#### 3.1 数据集

| 数据集 | 用途 | 规模 |
|--------|------|------|
| **MS MARCO Passage** | 主评估语料 | 从 880 万段落中筛选含人名的子集 |
| **MSMARCOFair** (Rekabsaz 2022) | 含性别标注的子集 | ~15K 段落 |
| **WinoBias** | 性别偏见标准测试集 | ~3K 句子 |

**方法**：
1. 从 MS MARCO 中用 NER 工具提取包含人名的真实段落。
2. 对这些段落进行**最小扰动替换**（只换名字，保留原始语法和内容）。
3. 在这些"半真实"数据上重复 TCD 分析，验证功能词 > 内容词是否依然成立。

#### 3.2 多模型对比

| 模型 | 架构 | 我们的角色 |
|------|------|----------|
| **ColBERTv2** | Multi-vector / Late Interaction | 主研究对象 — 可拆解 |
| **DPR** | Single-vector / Bi-encoder | 对照组 — 不可拆解 |
| **SPLADE** | Sparse Learned Weights | 另一种可解释架构 |
| **BM25** | 传统词频统计 | 非神经网络基线 |

**核心叙事不是"ColBERT 更有偏见"**，而是：ColBERT 是唯一能让你**看到偏见内部结构**的模型。其他模型可能同样有偏见（甚至更严重），但它们的单向量架构让你无法诊断。

> **ColBERT as a Diagnostic Instrument**：就像 X 光机不会让你生病，但它能让你看到病灶在哪里。

#### 3.3 资源需求

- **Phase 3.1**（MS MARCO 子集 + NER 筛选）：本地 Mac MPS，~4 小时
- **Phase 3.2**（多模型对比）：需要 1× GPU（16GB），~8 小时
- **建议方案**：使用学校 Oscar 集群

---

### Phase 4：算法干预（预计 2 周）

**核心问题**：既然偏见集中在功能词上，我们能不能只去偏功能词，而不损害检索精度？

#### 4.1 Null-Space Projection（零空间投影）

**思路**：
1. 收集所有反事实对中功能词的嵌入差异向量 δ = E(has|Emily) − E(has|Lakisha)。
2. 从这些 δ 中提取"身份敏感方向"（通过 PCA 或 SVD）。
3. 将所有功能词的嵌入投影到该方向的零空间中 — 即数学上"删除"身份信息。
4. 用修改后的嵌入重新计算 MaxSim，看偏见是否降低。

#### 4.2 效用-公平权衡（Utility-Fairness Tradeoff）

在去偏前后分别测量：
- **公平性**：SS, TCD 的下降幅度
- **检索质量**：MRR@10, NDCG@10, Recall@1000

**理想结果**：SS 下降 > 50%，MRR 下降 < 2%。这是顶会审稿人最看重的"actionable debiasing"。

---

### Phase 5：论文写作（预计 2 周）

#### 拟定标题

> **"Where Does Bias Hide? Token-Level Attribution Reveals Distributed Bias Propagation in Late-Interaction Retrieval"**

#### 贡献点清单（Contribution Statement）

一篇顶会论文需要 3-4 个清晰的贡献点：

1. **方法论贡献**：提出了 TCD（Token Contribution Disparity）指标，首次实现了对 Dense Retrieval 模型的 Token 级偏见归因。
2. **实证发现（发现 A）**：揭示了偏见在上下文化检索模型中通过功能词传导的"分布式传播"机制，推翻了传统的"词汇关联"偏见假设。
3. **实证发现（发现 B）**：发现了"罕见度中介效应"— BPE 分词器对少数族裔名字的碎片化处理放大了偏见，揭示了基础设施层面的公平性问题。
4. **应用贡献**：提出并验证了针对功能词的精准去偏方法（Null-Space Projection），在保持检索质量的前提下显著降低偏见。

#### 投稿目标

| 会议 | 截稿日期 | 匹配度 |
|------|---------|--------|
| **EMNLP 2026** | ~6 月 | ⭐⭐⭐⭐⭐ NLP + Fairness |
| **FAccT 2026** | ~1 月（已过，看 2027） | ⭐⭐⭐⭐⭐ 公平性专场 |
| **SIGIR 2026** | ~2 月（已过，看 Workshop） | ⭐⭐⭐⭐ IR 领域 |
| **NAACL 2026** | ~10 月 | ⭐⭐⭐⭐ NLP |
| **CIKM 2026** | ~5 月 | ⭐⭐⭐ IR + Data Mining |

**最佳目标：EMNLP 2026**（截稿约 6 月中旬，时间充裕）。

---

## Part III · 已知风险与缓解策略

| # | 风险 | 严重性 | 缓解策略 | 进度 |
|---|------|--------|---------|------|
| 1 | 合成数据生态效度 | 🔴高 | Phase 3 的 MS MARCO 验证 | 待做 |
| 2 | "Bias" 定义模糊 | 🟡中 | 采用 Counterfactual Fairness 框架 + 多层级指标 | 已规划 |
| 3 | 单模型贡献不足 | 🟡中 | Phase 3 的多模型对比 | 待做 |
| 4 | 只有可视化没有量化 | 🟢低 | 已有 TCD、Gini、ANOVA 等统计指标 | ✅ 已解决 |
| 5 | 去偏效果不确定 | 🟡中 | Phase 4 验证 | 待做 |
| 6 | GPU 资源 | 🟡中 | 学校 Oscar 集群 / Colab Pro | 可解决 |

---

## Part IV · 两个月时间线

| 周次 | 日期 | 任务 | 产出 |
|------|------|------|------|
| W1-2 | 3/10 – 3/23 | Phase 3.1：MS MARCO 数据预处理 + NER 筛选 | 真实反事实对数据集 |
| W3 | 3/24 – 3/30 | Phase 3.2：在真实数据上重复 TCD 分析 | "功能词效应在真实文本中是否成立" |
| W4 | 3/31 – 4/6 | Phase 3.3：多模型对比（DPR, SPLADE, BM25） | 跨模型偏见对比表 |
| W5 | 4/7 – 4/13 | Phase 4.1：Null-Space Projection 去偏实现 | 去偏算法代码 |
| W6 | 4/14 – 4/20 | Phase 4.2：Utility-Fairness Tradeoff 实验 | 效用-公平权衡曲线 |
| W7-8 | 4/21 – 5/4 | Phase 5：论文写作 + 图表制作 | 论文初稿 |

---

## 附录 · 实验代码与数据清单

| 文件 | 作用 | 数据量 |
|------|------|--------|
| `quick_experiment.py` | Phase 1 初步验证 | 8 对 |
| `expanded_experiment.py` | Phase 1 扩大规模 | 264 对 |
| `targeted_experiments.py` | Phase 1 完整统计检验 | 783 对 |
| `token_level_audit.py` | Phase 2 Token 级审计 | 12 对 |
| `function_word_bias.py` | Phase 2 功能词发现 | 650 对 |
| `rarity_profession_audit.py` | Phase 2 罕见度效应 | 1,140 组 |
| `finding_function_word_bias.md` | 发现 A 文档 | — |
| `finding_rarity_effect.md` | 发现 B 文档 | — |
