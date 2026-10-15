# MLLM / VLM 算法工程师：课程笔记与面试知识覆盖核对

核对日期：2026-10-14。这里把「公开岗位职责与任职要求」「论文/框架所界定的知识」「由此推测值得准备的面试追问」分开。**岗位要求不是面试题库**：不同公司、职级和团队会改变考察重点；下文的 P0/P1/P2 是学习优先级建议，不是对真实面试出现频率的统计结论。[腾讯招聘的多模态岗位](https://careers.tencent.com/jobdesc.html?postId=2048802520092016640)、[OpenAI 的多模态研究岗位](https://openai.com/careers/researcher-multimodal-safety-san-francisco/)、[Google DeepMind 招聘流程](https://deepmind.google/careers/)

## 结论先行

1. 现有 `notes.md` 已覆盖 Zero to Hero 从 Micrograd 到 Makemore Part 5 / WaveNet 的基础主线：计算图、字符模型、MLP、初始化和 BatchNorm、手动反向传播、层级序列模型。Karpathy 的课程页明确说，后续 GPT 和 tokenizer 是**另外的课**，因此不能把截至 WaveNet 的笔记称作完整 LLM 或 MLLM 课程。[Karpathy 课程大纲](https://karpathy.ai/zero-to-hero.html)
2. 维度变换值得重点练。PyTorch 官方接口直接以输入/输出形状定义 embedding 和 attention；LLaVA 原论文明确需要把视觉特征投影到语言 embedding 维度，才能交给语言模型。所以能解释 `C[Xb]`、展平、矩阵乘法、广播、视觉 token 拼接，是模型设计与调试的基础。**但“维度题必考/占多少比例”无法由这些一手资料证明。**[PyTorch Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)、[PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention)、[LLaVA §4.1](https://arxiv.org/html/2304.08485v2)
3. 就这几节课自身而言，当前笔记的基础概念和白板公式覆盖面已经有骨架，但还可加深重复索引的梯度汇聚、`ignore_index` / loss mask 的分母、BatchNorm 的训练/推理统计与小 batch 行为、梯度检查、数值稳定以及 WaveNet 感受野。这是对官方课程和框架行为的**备考推断**，不是“往年真题统计”。[Karpathy 课程大纲](https://karpathy.ai/zero-to-hero.html)、[PyTorch Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)、[PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)、[PyTorch BatchNorm1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)
4. 对完整 MLLM/VLM 岗位目标，**目前远未全面覆盖**：Transformer/GPT、tokenizer、ViT/视觉编码、CLIP 对比学习、视觉-语言连接与两阶段训练、数据/评测、微调与推理成本都需要后续系统学习。这个判断是把课程边界与公开岗位职责、原始论文逐项对照得出的。[Karpathy 课程大纲](https://karpathy.ai/zero-to-hero.html)、[腾讯招聘](https://careers.tencent.com/jobdesc.html?postId=2048802520092016640)、[OpenAI 多模态岗位](https://openai.com/careers/researcher-multimodal-safety-san-francisco/)

## 公开岗位到底强调什么

- 腾讯微信视觉的多模态大模型岗位写明参与模型**设计、训练、调优、评测**及电商、短视频、音乐内容理解落地；这支持把架构理解、实验与评估视为准备范围，却没有公布任何具体白板题。[腾讯招聘原页](https://careers.tencent.com/jobdesc.html?postId=2048802520092016640)
- OpenAI 的多模态研究岗位具体提到视觉编码器、投影层、模态融合、推理权衡、后训练、数据整理、评估与错误分析。这是本研究划定 VLM 知识图谱的重要证据；该岗位偏研究和安全，不可直接概括所有算法岗。[OpenAI 岗位原页](https://openai.com/careers/researcher-multimodal-safety-san-francisco/)
- OpenAI 的另一多模态感知工程岗位强调视觉/音频信号、Python/PyTorch、真实环境的鲁棒性、资源受限部署与系统集成。它说明岗位可能偏感知与工程，准备方向须跟目标 JD 调整。[OpenAI 多模态感知岗位](https://openai.com/careers/machine-learning-engineer-multimodal-perception-and-authentication-san-francisco/)
- Google DeepMind 把 Research Engineer 定义为兼具深度学习、数学与工程的实验型角色，并说明面试按具体岗位能力评估、各岗位流程略有不同。不能据此声称所有面试官“绝不考”完整反向传播代码。[Google DeepMind Careers](https://deepmind.google/careers/)

## 当前笔记与目标岗位的覆盖图

以下「覆盖程度」依据当前笔记的章节结构与官方课程大纲核对；「下一步」是学习建议，不是岗位方规定的题目。[当前笔记](notes.md)、[Karpathy 课程大纲](https://karpathy.ai/zero-to-hero.html)

| 知识域 | 当前程度 | 目标岗位为什么相关 / 下一步 | 一手来源 |
| --- | --- | --- | --- |
| 自动求导、梯度累加、数值梯度检查 | 已讲基础 | 白板解释计算图和参数为何更新；可继续练复杂分叉与梯度检查 | [Karpathy Micrograd](https://karpathy.ai/zero-to-hero.html) |
| Bigram、负对数似然、Embedding/MLP、维度 | 已讲基础 | 练真实张量形状、批维/特征维、重复索引反传；embedding 接口保留原输入形状并追加特征维 | [Karpathy Makemore](https://karpathy.ai/zero-to-hero.html)、[PyTorch Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html) |
| 初始化、激活饱和、BN、反向传播、WaveNet | 已讲基础 | 进一步核对 BN 的 batch 统计、running 统计和训练/推理区别，解释层级分组与感受野 | [Karpathy Part 3–5](https://karpathy.ai/zero-to-hero.html)、[PyTorch BN](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html) |
| Transformer/GPT 与 tokenizer | 未系统覆盖；只有 attention 形状桥接 | 学因果 mask、多头注意力、残差/LN、位置编码、next-token loss、BPE/tokenizer；课程后两讲专门覆盖 GPT 和 tokenizer | [Karpathy GPT、Tokenizer 大纲](https://karpathy.ai/zero-to-hero.html)、[Transformer 原论文](https://arxiv.org/abs/1706.03762) |
| 图像编码：ViT | 未系统覆盖 | 从图像 patch 到序列的形状、位置表示、分辨率改变如何影响 token 数量 | [ViT 原论文](https://arxiv.org/abs/2010.11929) |
| 图文对齐：CLIP | 未系统覆盖 | 图像/文本双编码器、batch 内配对的对比目标、与生成式 VLM 的目标区别 | [CLIP 原论文](https://arxiv.org/abs/2103.00020) |
| 视觉-语言连接与训练：LLaVA | 只讲了示意形状与 loss mask | 研究视觉编码器→投影层→语言模型、对齐预训练和视觉指令微调、冻结/解冻范围 | [LLaVA 原论文 §4](https://arxiv.org/html/2304.08485v2) |
| 动态分辨率和视觉 token 预算 | 未覆盖 | 分辨率如何影响视觉 token、上下文长度与成本；Qwen2-VL 明确采用可变数量视觉 token 的动态分辨率机制 | [Qwen2-VL 原论文](https://arxiv.org/abs/2409.12191) |
| 参数高效微调、数据和评估 | 未系统覆盖 | LoRA 冻结原权重并训练低秩增量；评测需区分感知、推理、图表/文档等能力 | [LoRA 原论文](https://arxiv.org/abs/2106.09685)、[MMMU 原论文](https://arxiv.org/abs/2311.16502)、[OpenAI 岗位](https://openai.com/careers/researcher-multimodal-safety-san-francisco/) |
| 推理/显存性能 | 只有融合与 FlashAttention 入门 | 掌握 attention 的 `T×T` 中间量、HBM/SRAM 数据搬运与分块；FlashAttention 是 IO-aware 的**精确** attention 实现，不能等同于“把 CE 合并求导” | [FlashAttention 原论文](https://arxiv.org/abs/2205.14135)、[PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention) |

### 一个值得避免的错觉

现有笔记第 7 节的 `(B,N,D_v) → (B,N,D_h)` 投影、拼接和 `QK^T` 形状练习**是桥梁，不是完整 LLaVA / Transformer 实现**。原始 LLaVA 论文还写明：阶段 1 冻结视觉编码器与 LLM，只训练投影矩阵；阶段 2 继续冻结视觉编码器，更新投影层与 LLM；训练时只对预测的回答 token 计算自回归损失。[LLaVA §4.1–4.2](https://arxiv.org/html/2304.08485v2)

## 仅围绕当前课程，建议补到每章后的追问

优先级是基于「掌握本节原理、能在白板上解释真实代码」的备考建议。公开 JD 和论文并没有给出这些题目的频率统计。[Google DeepMind 面试说明](https://deepmind.google/careers/)、[Karpathy 课程](https://karpathy.ai/zero-to-hero.html)

| 优先级 | 对应章节 | 值得自问的追问与回答边界 | 依据 |
| --- | --- | --- | --- |
| P0 | Micrograd | `x` 同时走两条路径，为何回传要 `+=`？给一个 `x*x+x` 数值例子；如何用有限差分核对而不把它当训练算法？ | [Karpathy Micrograd](https://karpathy.ai/zero-to-hero.html) |
| P0 | Bigram/MLP | `C:(V,D)`、`X:(B,K)`，`C[X]` 为什么是 `(B,K,D)`？展平只并入哪些轴？one-hot 乘矩阵与查表的关系？同一 ID 多次出现，梯度怎样累到同一行？ | [PyTorch Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)、[Karpathy MLP](https://karpathy.ai/zero-to-hero.html) |
| P0 | MLP/Backprop | `Y=XW+b` 的 `dX,dW,db` 及形状；广播过的 batch 轴为何在 `db` 中求和？维度检查是必要核对但不能取代链式法则推导。 | [Karpathy Backprop Ninja](https://karpathy.ai/zero-to-hero.html) |
| P0 | Backprop | Softmax+CE 为什么给出 `p-y`？batch mean、类别权重、label smoothing、`ignore_index` 改变了哪些条件？`F.cross_entropy` 的输入为什么是 logits？不要无条件把 `p-y` 原样套到加权/平滑/掩码情形。 | [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) |
| P0 | 训练诊断 | 均匀预测的初始 CE 为何约为 `log(V)`？`tanh` 饱和如何使梯度变小？BN 在 train/eval 各用什么统计量？ | [Karpathy Part 3](https://karpathy.ai/zero-to-hero.html)、[PyTorch BN](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html) |
| P1 | Bigram/MLP | `keepdim`、广播轴、`view/reshape/permute`、训练/验证/测试划分、学习率试验与恢复，能否用一个小张量说明？ | [Karpathy Makemore Part 2](https://karpathy.ai/zero-to-hero.html) |
| P1 | 训练诊断 | BN 的 `momentum` 与 SGD 动量为何不是同一件事？PyTorch 前向方差与 running variance 的估计约定有何不同？ | [PyTorch BN](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html) |
| P1 | WaveNet | 每次两两合并后 `(B,T,H)` 怎样变？一层能看见多少前文？教材里的树形分组与真正的因果空洞卷积哪里相似、哪里不同？ | [Karpathy Part 5](https://karpathy.ai/zero-to-hero.html) |
| P1 | 工程延伸 | 为什么稳定的 CE 实现接收 logits？为什么“解析化简求梯度”和“融合 GPU kernel/减少 HBM IO”相关但不等价？ | [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)、[FlashAttention 论文](https://arxiv.org/abs/2205.14135) |
| P2（特定岗位） | Backprop | 逐算子写完 BN 反传，或实现完整 Micrograd 引擎。这是学习课程的训练方式，但公开来源不支持“所有算法岗都会考”或“绝对不会考”的断言。 | [Karpathy 课程说明](https://karpathy.ai/zero-to-hero.html)、[Google DeepMind 岗位差异说明](https://deepmind.google/careers/) |

### 白板/手写内容如何排先后

在当前课程范围内，先能不看代码写出：`C[X]` 的逐轴形状、`Y=XW+b` 的三项梯度、普通 one-hot Softmax+CE 的 `p-y`（含 batch 平均）、一个分叉计算图、一个 loss mask 的有效位置计数。这是依据课程概念与 PyTorch 接口作出的**建议练习清单**，不是声称已经搜到统一的面试题库。[Karpathy 课程](https://karpathy.ai/zero-to-hero.html)、[PyTorch Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)、[PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

进入下一阶段后，再练 attention 的 `Q/K/V` 形状、`QK^T/√d`、因果 mask 与输出形状。PyTorch 官方 `scaled_dot_product_attention` 文档给出了这些轴、mask 语义及可选融合后端，可作为核对答案的基准。[PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention)

## 后续学习的推荐顺序

1. 完成 Karpathy GPT 与 tokenizer 两讲，把当前 `WaveNet → attention → GPT` 的过渡补实。课程大纲把这两讲列在 WaveNet 之后。[Karpathy 课程](https://karpathy.ai/zero-to-hero.html)
2. ViT → CLIP：先理解图像如何变成 patch 序列，再理解图文双编码器的对比学习；两篇原论文分别定义了这些核心做法。[ViT](https://arxiv.org/abs/2010.11929)、[CLIP](https://arxiv.org/abs/2103.00020)
3. LLaVA：连接器、两阶段训练、哪些权重冻结、回答 token 的监督范围；再对照 Qwen2-VL 理解动态分辨率与视觉 token 预算。不同 VLM 的具体实现不要混为一谈。[LLaVA](https://arxiv.org/html/2304.08485v2)、[Qwen2-VL](https://arxiv.org/abs/2409.12191)
4. LoRA、评估和系统性能：用原论文掌握低秩微调；以 MMMU 为例学习多模态评测维度；用 FlashAttention 区分数学复杂度、显存占用和 IO 开销。最后结合目标岗位 JD 决定视频/音频或部署专题的深度。[LoRA](https://arxiv.org/abs/2106.09685)、[MMMU](https://arxiv.org/abs/2311.16502)、[FlashAttention](https://arxiv.org/abs/2205.14135)、[腾讯招聘](https://careers.tencent.com/jobdesc.html?postId=2048802520092016640)
