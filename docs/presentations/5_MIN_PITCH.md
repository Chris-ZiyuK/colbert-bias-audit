# 5-Minute Pitch: Where Does Bias Hide?
*(ColBERT Bias Audit - 5分钟英文演示演讲稿及中文对照)*

**预计用时**: 4分30秒 - 5分钟
**适用场合**: 项目展示、汇报、电梯演讲 (Elevator Pitch)

---

## 1. The Hook & The Problem (0:00 - 1:00)

**[English]**
Imagine you're an employer typing a query into a recruitment platform: *"Who is a qualified doctor?"* 

You have two identical resumes. The only difference is the name at the top: one says "Emily," the other says "Lakisha." Decades of economics research shows that in human hiring, Emily gets 50% more callbacks. 

Today, AI search engines rank these resumes before a human ever sees them. So we asked: **Does the AI discriminate too? And more importantly, *where exactly* does the bias come from?** 

Most AI search models are black boxes—they give you a single score, but they don't tell you *why*. For this project, we audited a model called ColBERT. ColBERT is unique because it lets us see its calculations word-by-word. It allowed us to build an "X-ray machine" to see exactly which parts of a search query absorb bias.

**[中文]**
想象一下，你是一位雇主，正在招聘平台上输入搜索词：“谁是合格的医生？”

你有两份完全相同的简历，唯一的区别是顶部的名字：一份写着“Emily（常见白人女性名字）”，另一份写着“Lakisha（常见黑人女性名字）”。几十年的经济学研究表明，在人类的招聘中，Emily获得的面试机会要多出50%。

如今，在人类看到简历之前，AI搜索引擎就已经对它们进行了排序。所以我们提出了一个问题：**AI也会进行同样的歧视吗？更重要的是，这种偏见究竟是从哪里来的？**

大多数AI搜索模型都是黑盒——它们只给你一个总分，却不告诉你原因。在这个项目中，我们审查了一个名为ColBERT的模型。ColBERT的独特之处在于，它允许我们逐词查看其计算过程。这让我们能够建造一台“X光机”，准确地看到搜索语句中的哪些部分吸收了偏见。

---

## 2. Finding 1: The Function Word Anomaly (1:00 - 2:30)

**[English]**
Our first finding completely upended our expectations. 

When we swapped Emily for Lakisha in the document, we asked: *which words in the search query reacted the most?* You might guess the word "doctor," or the word "qualified." 

But nobody guessed the real answer: **the bias hides in the small words — words like "is", "a", and "who".** 

Why does this happen? The AI we use, BERT, constantly updates its understanding of every word based on context. Words like "doctor" have strong semantic anchors. Their meaning doesn't change easily. 

But grammatical function words—like "is"—have no intrinsic meaning. They are empty vessels. In the neural network, they act like **context sponges**. They absorb the most information from the surrounding candidate names, and therefore, they absorb the most bias. We found that these invisible grammatical words carry **1.44 times** more bias than meaningful words.

**[中文]**
我们的第一个发现完全颠覆了预期。

当我们在文档中把Emily换成Lakisha时，我们问：*搜索查询中哪些词的反应最强烈？* 你可能会猜是“doctor（医生）”或者“qualified（合格的）”。

但没人猜到真正的答案：**偏见隐藏在那些“小词”中——比如“is(是)”、“a(一个)”和“who(谁)”。**

为什么会这样？我们使用的AI（BERT）会不断根据上下文更新它对每个词的理解。像“医生”这样的词具有很强的语义锚点。它们的含义不容易改变。

但像“是”这样的语法功能词没有内在的含义。它们是空的容器。在神经网络中，它们就像**上下文的海绵**。它们从周围的候选人名字中吸收了最多的信息，因此，也吸收了最多的偏见。我们发现，这些看似隐形的语法词汇所携带的偏见，是有实意词汇的**1.44倍**。

---

## 3. Finding 2: The Tokenization Tax (2:30 - 3:30)

**[English]**
Second, we discovered a deep, structural problem we call the **Tokenization Tax**. 

Before an AI reads your name, it chops it into byte-sized pieces called tokens. Common names like "Emily" stay intact—they count as one piece. But rarer names are chopped up. "Lakisha" gets cut into three pieces: La-ki-sha. 

Here is the problem: rare names disproportionately belong to minority communities. And when the AI sees a severely fragmented name, it becomes mathematically uncertain. To compensate for this uncertainty, it falls back heavily on statistical stereotypes it learned from the internet. 

The result? Names that get chopped into many pieces suffer **1.84 times** more bias. This isn't just human prejudice; this is structural inequality built right into the AI's most basic plumbing.

**[中文]**
第二，我们发现了一个深刻的底层架构问题，我们称之为**分词税 (Tokenization Tax)**。

在AI读取你的名字之前，它会将其切分成名为Token（词元）的小块。像Emily这种常见名会保持完整——被算作一块。但生僻的名字会被切碎。“Lakisha”会被切成三块：La-ki-sha。

问题就在这里：生僻的名字往往不成比例地属于少数族裔群体。而当AI看到一个被严重碎片化的名字时，它在数学上会变得不确定。为了弥补这种不确定性，它会严重依赖于它从互联网上学到的统计偏见和刻板印象。

结果是什么？被过度切分的名字承受的偏见放大了**1.84倍**。这不仅仅是人类的偏见；这是直接刻在AI最基础的水管（底层逻辑）里的结构性不平等。

---

## 4. Finding 3: Decomposing Confounds (3:30 - 4:15)

**[English]**
Finally, we have to ask: what are we really measuring? 

Standard audits just swap names and call it a day, attributing everything to "racial bias." But replacing Emily with Lakisha changes many things at once: race, token fragments, and even socioeconomic signals. 

We ran over 48,000 controlled tests and used regression analysis to isolate each factor. We found that **gender mismatches and tokenization differences actually drive stronger and more consistent bias than race alone**. What classic studies often label simply as "racial discrimination" is actually a complex mixture of tokenization artifacts and gender noise in modern AI models. 

**[中文]**
最后，我们必须问：我们到底在测量什么？

标准的审查方法仅仅是交换名字就完事了，把一切都归咎于“种族偏见”。但用Lakisha替换Emily同时改变了许多变量：种族、分词碎片，甚至社会经济地位的信号。

我们运行了超过48,000次对照实验，并使用回归分析分离了每个因素。我们发现，**性别不匹配和分词差异实际上比单纯的种族因素驱动了更强、更一致的偏见。** 传统研究通常简单地标记为“种族歧视”的现象，在现代AI模型中，实际上是分词伪影和性别噪音的复杂混合体。

---

## 5. Conclusion (4:15 - 5:00)

**[English]**
In conclusion, bias in modern search engines doesn't hide where we thought it did. 

It is not localized in stereotypical words; it flows dynamically through the grammatical glue of the sentence. And it is amplified by something as invisible as how the AI chops up a word. 

If we want to build fair search algorithms, we cannot just try to "fix" the obvious words like doctor or nurse. We have to shine a light on the invisible infrastructure beneath the code. 

Thank you.

**[中文]**
总结一下，在现代搜索引擎中，偏见并没有藏在我们以为的地方。

它并不局限在具有刻板印象的实体词上；而是动态地流淌在句子的语法胶水中。而且，它会被像“AI如何切分单词”这种平时看不见的东西所放大。

如果我们想要构建公平的搜索算法，就不能仅仅去“修复”那些比如医生、护士等显然有偏见的词汇。我们必须用光照亮代码之下那些不可见的基础设施。

谢谢大家。
