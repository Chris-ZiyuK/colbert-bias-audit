# Publication Roadmap: ColBERT Bias Audit → Top Venue

## Part 1: Venue Analysis

### Target Venue Recommendation

| Venue | Fit | Deadline | Feasibility |
|-------|-----|----------|-------------|
| **EMNLP 2027** (via ARR) | ⭐⭐⭐⭐⭐ NLP + IR + fairness 核心赛道 | ARR May 2027 (预计) | 需补 3-4 个实验 |
| **FAccT 2027** | ⭐⭐⭐⭐ 审计方法论完美匹配 | Jan 2027 (预计) | 当前进度接近满足 |
| **SIGIR 2027** | ⭐⭐⭐ 需要 benchmark 数据 | ~Feb 2027 | 需 MS MARCO |
| **NAACL 2027** (via ARR) | ⭐⭐⭐⭐ 同 EMNLP | ARR cycle | 同 EMNLP |

> [!IMPORTANT]
> **推荐主投: EMNLP 2027 (via ARR)**
> - 理由：audience 最大、接受 Findings 也算发表、NLP fairness 是核心 track
> - 备选：FAccT 2027（如果实验做不完，FAccT 对 benchmark 要求低）
> 
> **EMNLP ARR deadline: 预计 May 2027 → 我们有约 12 个月准备**

---

## Part 2: Benchmark Papers Gap Analysis

### 5 篇标杆论文

| # | 论文 | 会议 | 与我们的关系 |
|---|------|------|------------|
| **P1** | Petrov et al. "Tokenizers Introduce Unfairness" | NeurIPS 2023 | 直接上游：tokenization→unfairness 范式 |
| **P2** | Manchanda & Shivaswamy "What is in a Name?" | ACL Findings 2025 | 最近竞品：name bias in embedding similarity |
| **P3** | Rekabsaz et al. "Societal Biases in Retrieved Contents" | SIGIR 2021 | IR bias 经典：aggregate-level audit |
| **P4** | Goldfarb-Tarrant et al. "MultiContrievers" | BlackboxNLP 2024 | 方法对标：probing dense retrieval representations |
| **P5** | Ovalle et al. "Tokenization Matters" | NAACL Findings 2024 | tokenization→fairness 在 NLP 的先例 |

### 与每篇论文的差距分析

#### vs P1 (Petrov — NeurIPS 2023)
| 维度 | Petrov | 我们 | Gap |
|------|--------|------|-----|
| 模型数量 | 17 tokenizers | 1 (ColBERTv2) | ❌ 严重不足 |
| 数据规模 | FLORES-200 (200 languages) | 65 names, synthetic | ❌ 需要 benchmark |
| 定量化 unfairness | cost, latency, context window | TCD score difference | ⚠️ 需要翻译成实际 ranking 影响 |
| 叙事 | "infrastructure audit" | 当前叙事模糊 | ⚠️ 叙事需调整 |

#### vs P2 (Manchanda — ACL 2025)
| 维度 | Manchanda | 我们 | Gap |
|------|-----------|------|-----|
| 模型数量 | 3 embedding models | 1 | ❌ 需要至少 2 个 |
| 干预/Mitigation | anonymization (已实现+评估) | 仅提出方向 | ❌ 最大差距 |
| 下游任务 | 3 NLP tasks validated | 0 | ❌ 需要 |
| Token-level分析 | ❌ 没有 | ✅ 我们的核心优势 | ✅ 独特贡献 |

#### vs P3 (Rekabsaz — SIGIR 2021)
| 维度 | Rekabsaz | 我们 | Gap |
|------|----------|------|-----|
| Benchmark | MS MARCO | synthetic + real-text | ❌ 需要 MS MARCO |
| Mitigation | adversarial mitigation (实现+评估) | ❌ 无 | ❌ |
| Analysis depth | aggregate-level | token-level | ✅ 更深入 |
| Multiple models | BERT rankers | ColBERT only | ⚠️ |

#### vs P4 (MultiContrievers — BlackboxNLP 2024)
| 维度 | MultiContrievers | 我们 | Gap |
|------|-----------------|------|-----|
| Probing 方法 | information-theoretic probing | TCD (directly decomposable) | ✅ 更 principled |
| 多次 seeds | 25 models, different seeds | 1 model | ❌ |
| Gender bias 分析 | extractability ≠ bias causation | confound decomposition | ✅ 因果更清晰 |

#### vs P5 (Ovalle — NAACL 2024)
| 维度 | Ovalle | 我们 | Gap |
|------|--------|------|-----|
| 问题具体性 | neopronouns 碎片化→misgendering | name 碎片化→score偏移 | ✅ 类似 |
| 干预 | pronoun parity (14%→58%) | ❌ 未实现 | ❌ 最大差距 |
| downstream 影响 | grammatical accuracy | retrieval score | ⚠️ 需要 ranking impact |

### Gap 综合排序（按严重程度）

| 优先级 | Gap | 影响 | 工作量 |
|--------|-----|------|--------|
| 🔴 **P0** | **缺少 mitigation 实验** | 所有标杆论文都有干预评估 | 3-4 周 |
| 🔴 **P1** | **单模型** | reviewer 一定会要求多模型 | 2-3 周 |
| 🟡 **P2** | **缺少 benchmark 验证** (MS MARCO) | EMNLP/SIGIR 必须，FAccT 不需要 | 2-3 周 |
| 🟡 **P3** | **ranking impact 缺失** | score→ranking 的翻译 | 1-2 周 |
| 🟢 **P4** | 叙事不够锋利 | 课程版可以，投稿版需要调整 | 1 周 |

---

## Part 3: Experiment Roadmap

### Phase A: Multi-Model Comparison（优先级 P1）
- **目标**：DPR (single-vector) 上做 aggregate-level bias 对比
- **方法**：用同样的 65 names × 33 professions 跑 DPR，对比 aggregate SS
- **预期**：DPR 有 bias 但无法做 token-level 归因 → 凸显 ColBERT 诊断价值
- **工作量**：2 周
- **实施**：`experiments/phase4_multi_model/dpr_comparison.py`

### Phase B: MS MARCO Counterfactual Evaluation（优先级 P2）
- **目标**：在真实 IR benchmark 上验证
- **方法**：NER-based name swapping on MS MARCO passages → re-rank → measure MRR/nDCG change
- **预期**：偏差在真实数据上也存在，且 rare names 表现更差
- **工作量**：3 周
- **实施**：`experiments/phase5_benchmark/msmarco_counterfactual.py`

### Phase C: Tokenizer Equalization Mitigation（优先级 P0）
- **目标**：实现并评估至少一种干预策略
- **方法**：
  1. 把 top-100 minority names 加入 BPE vocab（custom tokenizer）
  2. Re-encode 同样的 counterfactual pairs
  3. 对比 rarity amplification 是否下降
- **预期**：rarity amplification ratio 从 1.84× 下降
- **工作量**：2 周
- **实施**：`experiments/phase6_mitigation/tokenizer_equalization.py`

### Phase D: Ranking Impact Study（优先级 P3）
- **目标**：从 score difference 翻译到 ranking change
- **方法**：构建一个 100-doc 候选池，注入 counterfactual pairs，测 rank position change
- **预期**：rare names 的 rank drop 显著大于 common names
- **工作量**：1 周
- **实施**：`experiments/phase5_benchmark/ranking_impact.py`

### Timeline

```
2026 May-Jun:  Phase A (DPR comparison) + Phase D (ranking impact)
2026 Jul-Aug:  Phase B (MS MARCO counterfactual)
2026 Sep-Oct:  Phase C (mitigation)
2026 Nov-Dec:  Paper writing + narrative adjustment
2027 Jan:      FAccT backup submission (if ready)
2027 Mar-Apr:  Final polish
2027 May:      EMNLP ARR submission
```

---

## Part 4: Narrative Strategy (for publication version)

课程版叙事（当前）→ 投稿版叙事（目标）的关键调整：

| 课程版 | 投稿版 |
|--------|--------|
| "TCD 是核心贡献" | "Confound decomposition methodology + tokenization tax = 核心贡献" |
| "Function words carry more bias" | "Function-word TCD as a diagnostic substrate → enables rarity + decomposition discoveries" |
| "We did not implement mitigations" | "Tokenizer equalization reduces rarity amplification by X%" |
| 单模型分析 | "ColBERT enables token-level analysis; DPR confirms aggregate pattern; both show tokenization tax" |

标题调整（投稿版建议）：
> **"The Tokenization Tax: How BPE Fragmentation Amplifies Identity Bias in Dense Retrieval"**

---

## Part 5: Repository Restructuring

### Current Problems
1. 根目录有 HTML 展示文件（`project_showcase.html`, `formal_findings.html` 等）混在源码中
2. `paper/` 目录有过时的 main.tex 和 lit review（课程版在 `course/` 外部）
3. `experiments/` 命名不一致（phase0, phase1, phase2, phase2_real_text, phase3）
4. 缺少 `scripts/`（可复现脚本）、`configs/`（有但内容不明）
5. 课程驱动的文件（`5_MIN_PITCH.md`, `PRESENTATION.md`）和研究文件混在一起

### Target Structure

```
colbert-bias-audit/
├── README.md                          # 项目概述（重写 for research）
├── LICENSE
├── requirements.txt                   # 或 environment.yml
├── .gitignore
│
├── paper/                             # 投稿版论文
│   ├── main.tex
│   ├── refs.bib
│   ├── acl.sty
│   ├── figures/
│   └── appendix/
│
├── src/                               # 核心库代码
│   ├── __init__.py
│   ├── audit/
│   │   ├── core.py                    # TCD computation, MaxSim, etc.
│   │   ├── confound.py                # matched-pair families
│   │   └── names.py                   # name pool management
│   ├── models/
│   │   ├── colbert.py                 # ColBERT wrapper
│   │   └── dpr.py                     # DPR wrapper (Phase A)
│   └── utils/
│       ├── tokenizer.py               # BPE analysis utilities
│       └── stats.py                   # statistical tests
│
├── experiments/                       # 实验脚本（可复现）
│   ├── 01_control_validation/         # P0
│   ├── 02_function_word_bias/         # P1 initial
│   ├── 03_profession_expansion/       # P1 expansion
│   ├── 04_name_confound/              # P2/P2b
│   ├── 05_real_text_validation/       # RT
│   ├── 06_bm25_baseline/             # BM25
│   ├── 07_dpr_comparison/            # Phase A (new)
│   ├── 08_msmarco_counterfactual/    # Phase B (new)
│   ├── 09_ranking_impact/            # Phase D (new)
│   └── 10_mitigation/               # Phase C (new)
│
├── data/
│   ├── names/                         # name pools + annotations
│   │   ├── bertrand_mullainathan.json
│   │   └── rosenman_expanded.json
│   ├── professions/
│   │   └── professions.json
│   └── passages/                      # natural text passages
│       └── real_text_passages.json
│
├── results/                           # 实验输出（可能 .gitignore 大文件）
│   ├── 01_control/
│   ├── 02_function_word/
│   ├── ...
│   └── figures/                       # 论文用图（生成后存放）
│
├── scripts/                           # 一键复现脚本
│   ├── run_all.sh                     # 全流程复现
│   ├── generate_figures.py            # 论文图表生成
│   └── generate_tables.py            # 论文表格生成
│
├── docs/                              # 课程相关文件（归档）
│   ├── course_paper/                  # 课程版论文
│   │   ├── main.tex
│   │   └── refs.bib
│   ├── presentations/
│   ├── showcase/
│   └── progress/
│
└── configs/                           # 实验配置
    ├── colbert.yaml
    └── experiment.yaml
```

### Restructuring Actions

移动规则（不删除任何文件，只是重组）：

| 当前位置 | 目标位置 | 说明 |
|----------|----------|------|
| `5_MIN_PITCH.md` | `docs/presentations/` | 课程材料归档 |
| `PRESENTATION.md` | `docs/presentations/` | 课程材料归档 |
| `PROJECT_INTRODUCTION.md` | `docs/progress/` | 项目进展文档 |
| `AGENT_CONTEXT_P2_HANDOFF.md` | `docs/progress/` | 开发上下文 |
| `project_showcase.html` | `docs/showcase/` | 展示页面 |
| `project_overview.html` | `docs/showcase/` | 展示页面 |
| `formal_findings.html` | `docs/showcase/` | 展示页面 |
| `progress_tracker.html` | `docs/progress/` | 进度追踪 |
| `research_roadmap.md` | `docs/progress/` | 路线图 |
| `paper/` (旧) | `docs/drafts/` | 旧版草稿 |
| `presentation/` | `docs/presentations/` | 合并 |
| `notebooks/` | 保留（空的就删除 `.gitkeep`）| |
| 根目录 `course/` (外部) | `docs/course_paper/` | 课程版论文归档 |

---

## 需要确认的问题

> [!IMPORTANT]
> **Q1**: 主投 EMNLP (ARR, ~May 2027) 还是 FAccT (~Jan 2027)？这决定了实验优先级。
> - EMNLP: 需要全部 4 个 Phase（A-D），12 个月时间充足
> - FAccT: 只需 Phase A + C，8 个月更紧但可行
>
> **Q2**: 目录重组我现在立刻执行吗？会涉及大量 `git mv` 操作。
>
> **Q3**: DPR 对比实验你有 GPU 资源吗？Oscar HPC 可以用吗？
