# 深度学习底层核心复盘：从自动求导引擎到语言建模与多模态架构

本篇复盘记录了从标量级微积分自动求导引擎（Micrograd）出发，跨越到高维张量并行运算、语言模型建模（Makemore），直至现代多模态大模型（MLLM）工业级实现的完整认知演进链条。

---


## 目录

- [一、微观引擎篇：自动求导与计算图的本质 (Micrograd)](#一微观引擎篇自动求导与计算图的本质-micrograd)
  - [1. 节点拓扑：为什么是 DAG 而不是二叉树？](#1-节点拓扑为什么是-dag-而不是二叉树)
  - [2. 运算符重载与魔术方法](#2-运算符重载与魔术方法)
  - [3. 函数闭包与动态挂载：`_backward` 到底在执行什么？](#3-函数闭包与动态挂载_backward-到底在执行什么)
  - [4. 局部导数 vs 全局导数：梯度的传递源头](#4-局部导数-vs-全局导数梯度的传递源头)
  - [5. 核心考点：为什么梯度必须累加 (+=) 而不是赋值 (=)？](#5-核心考点为什么梯度必须累加--而不是赋值-)
  - [6. 拓扑排序 (Topological Sort) 的调度必要性](#6-拓扑排序-topological-sort-的调度必要性)
  - [7. 从标量神经元到多层感知机 (MLP) 的抽象流水线](#7-从标量神经元到多层感知机-mlp-的抽象流水线)
  - [面试延伸：计算图与自动求导](#interview-micrograd)
- [二、宏观跃迁篇：张量计算、广播机制与语言建模 (Makemore Bigram)](#二宏观跃迁篇张量计算广播机制与语言建模-makemore-bigram)
  - [1. 计算机数组 (Array) vs 数学矩阵 (Matrix)](#1-计算机数组-array-vs-数学矩阵-matrix)
  - [2. 统计学 Bigram 语言模型与滑动窗口配对](#2-统计学-bigram-语言模型与滑动窗口配对)
  - [3. 数据类型 (dtype) 与方法调用的常见陷阱](#3-数据类型-dtype-与方法调用的常见陷阱)
  - [4. 广播机制 (Broadcasting) 与 keepdim=True 深度解析](#4-广播机制-broadcasting-与-keepdimtrue-深度解析)
  - [5. 高维张量的几何直觉：活页记事本模型](#5-高维张量的几何直觉活页记事本模型)
  - [6. 文本生成采样：Multinomial 轮盘与 Generator 种子](#6-文本生成采样multinomial-轮盘与-generator-种子)
  - [面试延伸：张量、广播与采样](#interview-tensor-bigram)
- [三、范式跃迁篇：单层神经网络与交叉熵损失](#三范式跃迁篇单层神经网络与交叉熵损失)
  - [1. 单层线性网络的设计目的与数学等价性](#1-单层线性网络的设计目的与数学等价性)
  - [2. 独热编码 (One-Hot) 乘矩阵本质即查表 (Embedding Lookup)](#2-独热编码-one-hot-乘矩阵本质即查表-embedding-lookup)
  - [3. 从 Logits 到概率：Softmax 两步拆解](#3-从-logits-到概率softmax-两步拆解)
  - [4. 交叉熵损失函数解剖：高级整数索引 (Fancy Indexing)](#4-交叉熵损失函数解剖高级整数索引-fancy-indexing)
  - [5. 参数更新中的 .data 历史演进与现代规范](#5-参数更新中的-data-历史演进与现代规范)
  - [面试延伸：分类头与交叉熵](#interview-single-layer)
- [四、全局链路推演图](#四全局链路推演图)
  - [面试延伸：训练闭环](#interview-training-loop)
- [五、前沿映射篇：现代多模态大模型 (MLLM) 工业级落地](#五前沿映射篇现代多模态大模型-mllm-工业级落地)
  - [1. 算子融合与数值稳定：Log-Sum-Exp Trick](#1-算子融合与数值稳定log-sum-exp-trick)
  - [2. 多模态对齐投影层 (Multimodal Projector)](#2-多模态对齐投影层-multimodal-projector)
  - [3. 图文跨模态特征拼接与并行自注意力](#3-图文跨模态特征拼接与并行自注意力)
  - [面试延伸：多模态接口与高效计算](#interview-mllm-foundation)
- [六、正则化：统计学平滑 ↔ L2 权重衰减的对齐](#六正则化统计学平滑--l2-权重衰减的对齐)
  - [1. 统计学世界的危机：0 概率 → 无穷大损失](#1-统计学世界的危机0-概率--无穷大损失)
  - [2. 神经网络世界的对应物：L2 正则化](#2-神经网络世界的对应物l2-正则化权重衰减)
  - [3. 两者的对应关系](#3-顿悟两者的神圣对应)
  - [4. 工业界映射：AdamW 与标签平滑](#4-工业界映射大厂面试高频)
  - [面试延伸：正则化与泛化](#interview-regularization)
- [七、数据批处理：Mini-batch 机制与随机梯度下降 (SGD)](#mini-batch-sgd)
  - [1. 为什么不使用全量 Batch Gradient Descent？](#batch-gradient-descent)
  - [2. Mini-batch：无偏梯度估计与梯度噪声](#mini-batch-principle)
  - [面试延伸：批处理与梯度噪声](#interview-minibatch)
- [八、优化器生命线：学习率搜索与调度](#learning-rate)
  - [1. 学习率的作用与失效模式](#lr-failure-modes)
  - [2. 对数尺度的 LR Range Test](#lr-range-test)
  - [3. 学习率衰减](#lr-decay)
  - [面试延伸：学习率与训练稳定性](#interview-learning-rate)
- [九、MLP 字符语言模型：从上下文到交叉熵](#mlp-language-model)
  - [1. 前向数据流与核心公式](#mlp-forward)
  - [2. 张量形状索引](#mlp-shapes)
  - [3. 为什么使用 `F.cross_entropy`](#cross-entropy)
  - [面试延伸：Embedding 与语言模型](#interview-mlp-lm)
- [十、数据集划分与泛化诊断](#data-split)
  - [1. 训练、验证与测试集](#train-validation-test)
  - [2. 欠拟合、良好拟合与过拟合](#fit-diagnosis)
  - [面试延伸：数据治理与评估](#interview-data-split)
- [十一、从 MLP 到多模态大模型的训练映射](#mllm-training-map)
  - [面试延伸：MLLM 训练链路](#interview-mllm-training)
- [十二、让训练稳下来：初始化、归一化与动量](#stable-training)
  - [1. 输出层：先让模型“谦虚地猜”](#output-layer-init)
  - [2. 隐藏层：避开 `tanh` 的饱和区](#activation-saturation)
  - [3. 不再凭感觉缩放：Xavier 与 Kaiming 初始化](#xavier-kaiming-init)
  - [4. BatchNorm：训练过程中的“实时校准”](#batchnorm)
  - [5. Transformer 为什么更偏爱 LayerNorm / RMSNorm](#ln-rmsnorm)
  - [6. 动量到底是什么：给更新方向加上惯性](#optimizer-momentum)
  - [7. 两种 Momentum，别混淆](#two-momentums)
  - [面试延伸：初始化、归一化与动量](#interview-stable-training)
- [十三、Karpathy 反向传播课：从手推计算图到融合交叉熵](#backprop-ninja)
  - [1. 为什么要把一行交叉熵拆开？](#backprop-why-unpack)
  - [2. 一次前向传播中，数据依次经历了什么？](#backprop-forward-pipeline)
  - [3. 反向传播传递的到底是什么？](#backprop-what-is-gradient)
  - [4. 遇到不同算子时，梯度怎样分配？](#backprop-rules)
  - [5. 交叉熵如何一路反推回 logits？](#backprop-cross-entropy)
  - [6. logits 如何继续反推回 embedding 表？](#backprop-mlp-chain)
  - [7. 手推梯度的结果如何验证？](#backprop-checking)
  - [8. 为什么推导完成后反而要使用融合算子？](#backprop-fusion)
  - [9. MLLM 算法岗面试：会问什么，怎样回答？](#backprop-interview)

---

## 一、微观引擎篇：自动求导与计算图的本质 (Micrograd)

深度学习框架的核心使命：**利用前向传播记录计算轨迹，通过反向传播自动计算每个参数对最终损失的导数，进而沿负梯度方向更新参数。**

```text
【微观求导逻辑闭环】

前向计算:  a, b ──(算子: +, *)──> out (记录 _prev, 闭包定制 _backward)
    │
    ▼
拓扑排序:  从最终标量 Loss 出发，DFS 逆向梳理所有依赖节点
    │
    ▼
反向点火:  Loss.grad = 1.0 ──(按拓扑逆序触发 _backward)──> 叶子节点累加梯度 (+=)
    │
    ▼
参数更新:  W.data += -lr * W.grad (沿负梯度下降，逼近最优解)
```

### 1. 节点拓扑：为什么是 DAG 而不是二叉树？

- 二叉树结构中，任意子节点仅能拥有一个唯一的父节点。
- 深度学习计算图中，一个变量可以同时作为多个下游计算路径的输入（例如 $y = x \cdot x + x$ 中，x 同时参与了乘法与加法）。
- 计算图的真实数据结构是 **DAG（Directed Acyclic Graph，有向无环图）**。
- 代码中使用 `self._prev = set(_children)`：
  - 利用 `set` 集合自动去重（即使变量被复用多次，也只保留唯一的上游引用）；
  - 提供 $O(1)$ 复杂度的查找速度，快速追踪前驱依赖。

### 2. 运算符重载与魔术方法

- **运算符重载（Operator Overloading）**：Python 解释器在执行中缀算式（如 `a + b`）时，会自动将其转译为底层魔术方法调用 `a.__add__(b)`。同理，`a * b` 转译为 `a.__mul__(b)`。
- **参数位置与绑定**：
  - 运算符左侧的操作数绑定为实例自身（`self`）；
  - 运算符右侧的操作数作为形参传入（`other`）。
- **`__repr__` 规范化打印**：如果不重写 `__repr__`，打印实例仅能得到对象的内存物理地址；重写后输出类似 `Value(data=2.0, grad=0.0)`，极大降低调试排错成本。

### 3. 函数闭包与动态挂载：`_backward` 到底在执行什么？

在类的初始化中定义了 `self._backward = lambda: None`，为何最终调用 `loss.backward()` 却能层层传导梯度？

核心机制是 **Python 局部作用域的函数闭包（Closure）与对象属性动态覆盖**：

1. **叶子节点的保底安全设计**：  
   原始输入 $x$ 和模型权重 $w$ 属于叶子节点（Leaf Nodes），没有上游加工原料（`_prev` 为空），无须再向前传递梯度。默认赋予一个空操作 `lambda: None`，保证后续批量循环调用 `v._backward()` 时安全跳过，避免因缺少方法而抛出异常。
2. **算子内的现场捕获**：  
   在执行 `__mul__` 等具体算子时，当前函数环境唯一知晓：运算类型（乘法）、输入原料（`self` 和 `other`）、计算产物（`out`）。在函数内部就地定义局部函数：
   ```python
   def _backward():
       self.grad += other.data * out.grad
       other.grad += self.data * out.grad
   out._backward = _backward
   ```
   闭包机制将当前作用域中的 `self`、`other`、`out` 连同求导公式一同封存打包。
3. **动态属性替换**：新生成的节点 `out`，其刚初始化的空函数 `lambda: None` 被直接覆盖为其专属的定制反向逻辑。当反向传播逆序遍历到该节点时，触发的正是此定制函数。

### 4. 局部导数 vs 全局导数：梯度的传递源头

- **局部导数（Local Gradient）**：当前算子直接输出关于直接输入的偏导数。例如在乘法算子中：

  $$
  \frac{\partial out}{\partial a} = b.data
  $$

  （对应代码里的 `other.data`）。
- **全局导数（Global Gradient）**：整个网络的最终标量损失 $L$ 关于当前节点输出的偏导数 $\frac{\partial L}{\partial out}$（对应代码中的 `out.grad`）。
- **链式法则传递**：

$$
\frac{\partial L}{\partial a} = \frac{\partial L}{\partial out} \cdot \frac{\partial out}{\partial a}
$$

- **全局梯度的源头点火**：在调用 `loss.backward()` 时，启动代码必须显式设定：

```python
self.grad = 1.0  # 最终标量对自身的偏导数恒等于 1.0 (dL/dL = 1)
```

该初始值 `1.0` 作为全局梯度的始发动力，通过拓扑序列自顶向下依次与局部导数相乘并向前回传。

### 5. 核心考点：为什么梯度必须累加 (+=) 而不是赋值 (=)？

若在代码中写为 `self.grad = ...`，在遇到变量复用（如多元函数 $y = x + x$）时：

- **数学理论推导**：

  $$
  \frac{dy}{dx} = 1 + 1 = 2
  $$
- **若采用赋值操作 `=`**，后一条反向路径的计算结果将直接覆盖前一条路径的计算结果，最终错误得出 $\frac{dy}{dx} = 1$；
- **根据多元微积分链式法则**，自变量对因变量的总贡献必须是所有汇聚路径的偏导数代数累加和；
- **PyTorch 映射考点**：这也解释了为何在 PyTorch 训练循环中，每次迭代前必须调用 `optimizer.zero_grad()`。PyTorch 底层张量计算同样遵循累加机制，若不清空，历史批次的梯度将残留在内存中污染当前轮次的参数更新。

### 6. 拓扑排序 (Topological Sort) 的调度必要性

反向传播不能简单使用列表做无序遍历：

- 假设节点 $A$ 的输出分流给了 $B$ 和 $C$，而 $B$ 和 $C$ 又汇聚到了最终损失 $L$；
- 计算 $\frac{\partial L}{\partial A}$ 必须依赖完整的 $\frac{\partial L}{\partial B}$ 与 $\frac{\partial L}{\partial C}$；
- 拓扑排序通过深度优先搜索（DFS）后序遍历，确保所有依赖子节点被访问完毕后才推入序列；
- 其逆序 `reversed(topo)` 严格保证：任何计算节点在自身反向传播前，其下游的所有消费节点均已完成反向结算。

### 7. 从标量神经元到多层感知机 (MLP) 的抽象流水线

- **Neuron(nin)**：配置 `nin` 个可学习权重 `w` 与 1 个偏置 `b`。前向执行线性组合后接入非线性激活：

$$
\text{out} = \text{ReLU}\left(\sum_{i=1}^{nin} w_i x_i + b\right)
$$

- **Layer(nin, nout)**：单层包含 `nout` 个独立的神经元。相同的输入向量 $x$ 分发给所有神经元并行打分，组装成长度为 `nout` 的输出特征列表。
- **MLP(nin, nouts)**：流水线级联组织。通过列表拼接构建通道序列 `sz = [nin] + nouts`，相邻通道两两配对生成层级实例，实现前一层的输出透明输入下一层。

<a id="interview-micrograd"></a>

### 面试延伸：计算图与自动求导

**问题 1：为什么反向传播的计算图是 DAG，而不是树？**

**回答：** 同一中间变量可以被多个后续算子复用，例如 $y=x^2+x$ 中的 $x$ 同时走乘法和加法路径；树不允许一个子节点有多个父节点，DAG 可以。共享节点意味着其梯度必须汇总所有下游路径的贡献。

**问题 2：为什么梯度用 `+=` 而不是 `=`？**

**回答：** 根据多元链式法则，变量对 loss 的总影响是所有路径导数之和。若覆盖赋值，后一条路径会抹掉前一条贡献；PyTorch 默认累加 `.grad`，因此每次新迭代前必须 `zero_grad()` 或设为 `None`。

**问题 3：拓扑排序在反向传播中保证了什么？**

**回答：** 它保证某节点执行反传前，所有依赖该节点的下游节点已经把梯度传回，因此该节点拿到的是完整上游梯度。实现上通常对前向 DAG 做 DFS 后序遍历，再按逆序执行反传。

---

## 二、宏观跃迁篇：张量计算、广播机制与语言建模 (Makemore Bigram)

标量级自动求导能够穿透计算原理，但无法支撑工业级并发需求。现代深度学习必须依赖张量（Tensor）与底层 GPU 并行计算。

```text
【统计表到张量的演进】

文本字符对 (ch1, ch2) ──> 计数统计矩阵 N (27, 27) ──(dtype 转换)──> float 矩阵
                                                                      │
                                                                      ▼
概率矩阵 P ◄──(除法拉伸对齐)── P.sum(1, keepdim=True) 广播列向量 (27, 1)
     │
     ▼
采样推理: torch.multinomial(p) 掷骰子 ──> 获取下一个 Token 索引 ──> 解码吐词
```

### 1. 计算机数组 (Array) vs 数学矩阵 (Matrix)

| 比较维度 | 计算机数组 (Array)       | 数学矩阵 (Matrix)          | PyTorch 张量 (Tensor)  |
| ---- | ------------------- | ---------------------- | -------------------- |
| 本质定位 | 内存数据结构（连续物理存储）      | 空间代数算子（线性映射规则）         | 统一的通用多维张量容器          |
| 维度限制 | 自由，支持 1D、2D、3D 直至高维 | 必须且只能是二维（行 $\times$ 列） | 0D（标量）到高阶连续分布        |
| 默认乘法 | 逐元素对应相乘 (A * B)     | 行列内积投影 (A @ B)         | 支持 `*` 逐元素与 `@` 矩阵乘法 |

### 2. 统计学 Bigram 语言模型与滑动窗口配对

- **形式化任务**：基于当前字符预测紧邻的下一个字符，即估计条件概率分布 $P(w_t \mid w_{t-1})$。
- **字符映射表**：建立双向映射 `s_to_i`（String to Index 编码器）与 `i_to_s`（Index to String 解码器）。
- **避坑要点**：必须先在 `s_to_i` 中把全部特殊字符（如起始/结束标志符 `'.'`）注册完毕，之后再遍历 items 生成 `i_to_s`。若先生成反向映射再追加特殊符号，会导致解码字典缺失索引，引发 `KeyError`。
- **错位拉链提取**：通过 `zip(w, w[1:])` 实现序列错位切片：
  - `w` 对应时刻 $[t]$；
  - `w[1:]` 对应时刻 $[t+1]$；
  - `zip` 自动根据短序列截断，实现无缝抽取监督样本对 `(x, y)`。

### 3. 数据类型 (dtype) 与方法调用的常见陷阱

- **统计频数选用 `torch.int32`**：整型在内存中存储紧凑，在密集自增累加 `N[ix1, ix2] += 1` 时不会产生浮点舍入误差。
- **归一化概率必须转为 `.float()`**：整型张量无法承载实数除法，必须转换为 `torch.float32`。
- **语法陷阱辨析**：`tensor.float` 返回的是方法本身的函数指针；必须显式加括号 `tensor.float()` 调用才能返回转换后的数据对象，否则后续调用 `.sum()` 将抛出 `AttributeError`。


### 4. 广播机制 (Broadcasting) 与 keepdim=True 深度解析

- **广播基本原则**：从右向左对齐（Right-to-Left Alignment）。两张量进行代数运算时，维度从最右端（末尾维度）开始向左侧依次比对：
  - 当前维度的长度严格相等；
  - 或者其中一个张量在该维度的长度为 1（可沿此轴拉伸复制）；
  - 或者其中一个张量在该维度缺失（高位自动补 1）。

#### 实验复盘：矩阵行归一化

设有一个 $3 \times 3$ 的频数矩阵 $P$：

```text
矩阵 P (Shape: (3, 3)):
第 0 行: [   1,    2,    3 ]  --> 行和 = 6
第 1 行: [  10,   20,   30 ]  --> 行和 = 60
第 2 行: [ 100,  200,  300 ]  --> 行和 = 600
```

**错误案例：丢失维度的 `P.sum(1)`**

未保留维度时，求和结果降维为一维向量，Shape 退化为 `torch.Size([3])`。执行 `P / row_sums` 时进行对齐：

```text
P 的形状:         ( 3 ,  3 )
row_sums 的形状:        ( 3 )  <-- 自动被视作 (1, 3) 行向量！
```

最右端的 3 发生对齐。PyTorch 将其视作行向量，在运算时向下垂直复制 3 行：

```text
[ 6,  60,  600 ]
[ 6,  60,  600 ]
[ 6,  60,  600 ]
```

**后果**：矩阵每一列的元素被除以了同一个数，原本要计算的行归一化被静默错算为列归一化，由于计算维度完全兼容，代码不报错但业务逻辑全毁。

**正确案例：保留维度的 `P.sum(1, keepdim=True)`**

保留被压缩的维度，Shape 严格维持为 `torch.Size([3, 1])`。对齐过程：

```text
P 的形状:         ( 3 ,  3 )
row_sums 的形状:  ( 3 ,  1 )  <-- 锁定为 3 行 1 列的列向量
```

最右端的 1 匹配到 3，触发向右水平拉伸复制 3 次：

```text
[   6,    6,    6 ]
[  60,   60,   60 ]
[ 600,  600,  600 ]
```

矩阵每一行准确除以该行自己的总和，每行元素之和严格等于 1.0。

### 5. 高维张量的几何直觉：活页记事本模型

对于难以直观想象的三维张量 `(D0, D1, D2)`，建立如下空间映射：

- $D_0$（最外层）：记事本装订的总页数（Pages / Batches）；
- $D_1$（中间层）：每一页上记录的总行数（Rows / Sequence Length）；
- $D_2$（最内层）：每一行包含的特征格子数（Cols / Hidden Dimension）。

**空间拉伸推演：A(5, 1, 4) 与 B(3, 4)**

- 张量 A：一叠装订了 5 页的活页本，但每一页只有 1 行长条（长 4 格）；
- 张量 B：高位补 1 视作 `(1, 3, 4)`，是一张未装订的单页纸，上面写了 3 行 4 列；
- 空间运算 `A + B`：
  - **页面内部**：张量 A 在每页内部将唯一的 1 行向下纵向复制 3 份，铺满单页（1 变 3）；
  - **页面外部**：张量 B 这一页单纸被全自动复印 5 份，插入到 5 个页码中（1 变 5）；
  - 最终在三维空间中对齐为 `(5, 3, 4)` 的完整立体数据块。

### 6. 文本生成采样：Multinomial 轮盘与 Generator 种子

- **`torch.Generator().manual_seed(seed)`**：设置伪随机数发生器的随机种子，确保模型采样结果在跨平台、跨环境下的严格可复现性。
- **`torch.multinomial(p, num_samples=1, replacement=True)`**：大模型生成输出的基础采样算子。依据概率分布向量构建旋转轮盘，数值高的 Token 拥有更大的抽中面积。`replacement=True` 保证每次采样彼此独立。

<a id="interview-tensor-bigram"></a>

### 面试延伸：张量、广播与采样

**问题 1：`keepdim=True` 为什么常见？**

**回答：** 它保留被归约的维度，使归约结果能按预期广播回原张量。例如 logits 为 `(B, V)`，沿 `V` 求和后保留为 `(B, 1)`，才能对每一行做归一化；若变成 `(B,)`，会从尾部对齐，语义可能静默出错。

**问题 2：`*` 与 `@` 在 PyTorch 中分别表示什么？**

**回答：** `*` 是逐元素乘法，需要张量形状可广播；`@` 是矩阵乘法，最后两维按线性代数规则相乘，前导维可作为 batch 维广播。Transformer 的线性投影和 attention score 都依赖后者。

**问题 3：生成时为什么不总是选概率最大的 token？**

**回答：** 贪心解码稳定但容易重复、缺乏多样性；`multinomial` 按概率采样，保留低概率但合理候选的机会。实际 LLM 常结合 temperature、top-k 或 top-p 控制质量和多样性的权衡。

---

## 三、范式跃迁篇：单层神经网络与交叉熵损失

```text
【神经网络前向与反向流水线】

字符索引 xs ──> One-Hot 编码 (N, 27) ──> 线性层: xenc @ W ──> 未归一化打分 Logits (N, 27)
                                                                        │
                                                                        ▼
高级索引挑选正确项概率 ◄── 归一化概率 Probs (N, 27) ◄── 沿行归一化 ◄── Logits.exp() (软最大化)
     │
     ▼
probs[torch.arange(N), ys] ──> 取对数 .log() ──> 算术平均 .mean() ──> 乘负号取反 ──> Loss
                                                                        │
                                                                        ▼
W.data += -lr * W.grad ◄── W.grad 累加反传完毕 ◄── loss.backward() 触发拓扑图回溯
```

### 1. 单层线性网络的设计目的与数学等价性

- **设计意图**：输入为 27 维字符标识，输出目标为 27 维下一个字符分类，最简网络结构即为一个无偏置的单一线性权重矩阵：

$$
W \in \mathbb{R}^{27 \times 27}
$$
- **前向传播公式**：

$$
\text{Logits} = X_{\text{one-hot}} @ W
$$

- **数学等价性**：该单层模型经梯度下降收敛后，其权重参数矩阵 $W$ 经过 Softmax 变换后，在数值上将无限逼近纯统计学方法统计出的真实联合概率矩阵 $P$。这证明了神经网络能够自主拟合出数据底层的先验分布。

### 2. 独热编码 (One-Hot) 乘矩阵本质即查表 (Embedding Lookup)

- **矩阵运算本质**：一个只有第 $k$ 位为 1 的独热行向量乘以矩阵 $W$，计算结果恒等于矩阵 $W$ 的第 $k$ 行切片。
- **工业级实现转换**：若直接采用 One-Hot 编码，在超大词表（如 100k 规模）下，矩阵不仅稀疏（99.999% 的元素为 0），还会瞬间挤爆显存带宽；现代框架（如 `nn.Embedding`）在底层通过显存直接物理寻址取代稀疏矩阵乘法，以 $O(1)$ 时间复杂度直接抓取对应行的连续特征向量。

### 3. 从 Logits 到概率：Softmax 两步拆解

线性层直接输出的 Logits 处于实数域 $(-\infty, +\infty)$，将其转化为严谨概率分布必须经历 Softmax 操作：

1. **取指数（`logits.exp()`）**：将实数映射至严格正数区间 $(0, +\infty)$；通过指数级放大效应拉大极值差距，强化置信度。
2. **行归一化（除以 `sum(1, keepdim=True)`）**：结合广播机制，使每一行元素之和严格缩放至 1.0。

### 4. 交叉熵损失函数解剖：高级整数索引 (Fancy Indexing)

计算负对数似然损失的核心切片操作：

```python
loss = -probs[torch.arange(N), ys].log().mean()
```

- **高级整数索引运作方式**：`torch.arange(N)` 传入行序列，真实标签 `ys` 传入列序列。PyTorch 沿两轴坐标配对，精准提取出网络为真实标签所分配的预测概率值 $P(y_i)$。
- **负对数似然（NLL Loss）逻辑**：
  - 最大化真实样本的联合出现概率（最大似然估计）：

    $$
    \prod_i P(y_i)
    $$

  - 取自然对数将乘积转换为加法，避免浮点数下溢：

    $$
    \sum_i \log P(y_i)
    $$
  - 加上负号并将求和转为求平均（`.mean()`），适配梯度下降算法最小化损失的目标。

### 5. 参数更新中的 .data 历史演进与现代规范

- **历史原因**：早期 PyTorch 中 `tensor.data` 是绕开计算图追踪、就地修改显存底层的直接手段。
- **现代规范**：直接访问 `.data` 存在安全隐患，会破坏 Autograd 的版本依赖检测机制。工业级实现必须采用无梯度上下文管理器或封装优化器：

```python
# 规范方式一：显式上下文管理
with torch.no_grad():
    W -= lr * W.grad

# 规范方式二：优化器托管
optimizer.step()
```

<a id="interview-single-layer"></a>

### 面试延伸：分类头与交叉熵

**问题 1：为什么分类头输出 logits，而不是直接输出概率？**

**回答：** logits 处在无约束实数空间，便于线性层表达任意相对偏好；框架将 softmax 与交叉熵融合为数值稳定的 `logsumexp` 计算。若先手动 softmax 再传给 `CrossEntropyLoss`，会重复归一化并损失稳定性。

**问题 2：One-Hot 乘权重矩阵为什么等价于 embedding lookup？**

**回答：** One-Hot 只有一个位置为 1，乘矩阵时只会选中对应行；`nn.Embedding` 直接取该行，避免构造大而稀疏的 One-Hot 张量。词表很大时，查表在显存和带宽上都远优于稀疏乘法。

**问题 3：为什么参数更新不推荐 `.data`？**

**回答：** `.data` 会绕过 Autograd 的版本追踪，可能破坏需要保存张量的反向图且不会及时报错。优化器的 `step()` 或 `torch.no_grad()` 块会明确表达“这是不参与求导的参数更新”，更安全。

---

## 四、全局链路推演图

```text
1. 目标设定: 学习序列数据中 token_t 到 token_{t+1} 的转移规律
   │
   ▼
2. 统计学基线: 统计 27x27 频数表并归一化 (遇到上下文维度灾难)
   │
   ▼
3. 破局思路: 引入可微分参数矩阵替代死板查表，通过梯度下降自适应拟合
   │
   ▼
4. 底层基建 (Micrograd):
   - 构建 DAG 计算图与闭包属性挂载
   - 依赖拓扑排序确定反向传播时序
   - 链式法则与局部导数/全局导数相乘并累加 (+=)
   │
   ▼
5. 性能跃迁 (Makemore):
   - 从标量计算升级到多维张量 (Tensor)
   - 攻克广播机制 (Broadcasting) 与 keepdim 维度对齐
   │
   ▼
6. 神经网络闭环:
   One-Hot / Embedding ──> 线性投影 ──> Softmax 归一化
   ──> 交叉熵损失 (Fancy Indexing) ──> 自动求导与参数更新
```

<a id="interview-training-loop"></a>

### 面试延伸：训练闭环

**问题 1：请从输入到更新，描述一次训练 step。**

**回答：** 取一个 batch 后前向计算 logits 和 loss；清空旧梯度；`loss.backward()` 按计算图得到参数梯度；优化器按学习率、动量和权重衰减更新参数。训练模式下还可能包含 dropout、数据增强和梯度裁剪。

**问题 2：loss 下降是否说明模型一定变好？**

**回答：** 只说明当前优化目标在当前数据上降低。训练 loss 下降而验证 loss 上升是过拟合信号；还要结合任务指标、验证集表现、数据分布和数值稳定性判断。MLLM 还必须防范图文数据泄漏和评测污染。

---

## 五、前沿映射篇：现代多模态大模型 (MLLM) 工业级落地

上述微观基础在当今大语言模型（LLaMA-3, Qwen-2）与多模态大模型（LLaVA, CLIP）中，被直接映射为核心生产模块：

### 1. 算子融合与数值稳定：Log-Sum-Exp Trick

在工业级大模型训练框架中，绝不手动拆分编写 `.exp() -> sum -> .log()`：

- **数值溢出隐患**：若 Logit 数值超过极限（如大于 88.7），`exp` 操作会发生数值上溢（Overflow）输出 `inf`；若极低则发生下溢（Underflow）导致 `log(0)` 输出 `NaN`，引发训练中断。
- **底层融合算子**：直接使用 `torch.nn.functional.cross_entropy`。该算子在底层 CUDA 核心内实现 Log-Sum-Exp 优化：

$$
\log\left(\sum_{j} e^{z_j}\right) = c + \log\left(\sum_{j} e^{z_j - c}\right), \quad \text{其中 } c = \max(z)
$$

通过预先减去行内最大值使指数项维持在数值稳定区间，同时减少全局显存（HBM）的访存读写次数。

### 2. 多模态对齐投影层 (Multimodal Projector)

在典型多模态大模型（如 LLaVA 系列）中，连接视觉与文本的架构就是一个标准的线性层或多层感知机：

```python
# LLaVA 架构中的特征投影层定义
class MultimodalProjector(nn.Module):
    def __init__(self, visual_dim=1024, text_dim=4096):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(visual_dim, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, text_dim)
        )

    def forward(self, x):
        return self.proj(x)
```

- **物理意义**：视觉编码器（ViT）产出图像块特征张量 `(Batch, Num_Patches, 1024)`；投影层通过矩阵乘法，将视觉特征由 1024 维空间线性旋转并投影至文本模型能够理解的 4096 维词嵌入空间中。

### 3. 图文跨模态特征拼接与并行自注意力

当图像与文本分别完成空间对齐后，现代多模态框架会在序列维度（Dimension 1）直接执行张量拼接：

```python
# 视觉与文本 Token 在序列维度拼接融合
inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
# 拼接后的张量形状: (Batch, Seq_Len_Img + Seq_Len_Text, Hidden_Dim)
```

此后，多头自注意力机制（Multi-Head Attention）与位置编码加法广播机制，将无差别地覆盖在这组混合序列张量之上。计算底层仅存在统一步调的张量并行计算。

<a id="interview-mllm-foundation"></a>

### 面试延伸：多模态接口与高效计算

**问题 1：视觉 encoder 输出如何接入 LLM？projector 只是在“改维度”吗？**

**回答：** ViT 先把图像转为 patch-level 视觉 token，projector 再将其从视觉 hidden size 映射至 LLM hidden size，之后与文本 embedding 拼接或通过 cross-attention 交互。它不只是形状适配层，还是可学习的跨模态语义对齐接口；训练时图文监督会更新 projector，使视觉特征变得可被语言模型利用。

**问题 2：图文 token 拼接与 cross-attention 有什么区别？**

**回答：** 拼接后用同一个 self-attention 处理所有 token，实现简单但序列变长、注意力成本随长度平方增长；cross-attention 让文本查询视觉 token，可控制交互位置并适合部分 encoder-decoder 架构。选择取决于模型骨干、token 数量、是否需要冻结 LLM 和计算预算。

**问题 3：Log-Sum-Exp 与 FlashAttention 解决的是同一类问题吗？**

**回答：** 两者都重视数值稳定和减少中间数据，但层级不同。Log-Sum-Exp 稳定计算 softmax / 交叉熵；FlashAttention 则对 Attention 按块组织计算，避免物化完整注意力矩阵并减少 HBM 访问。前者是稳定公式，后者是结合内存层级的高效算法与内核实现。

---

## 六、正则化：统计学平滑 ↔ L2 权重衰减的对齐

### 1. 统计学世界的危机：0 概率 → 无穷大损失

**问题**：回到 $27 \times 27$ 频数矩阵 $N$。若训练集里 `q` 后面永远只跟 `u`，则 $N[q, x] = 0$，算得 $P(x \mid q) = 0.0$。

测试集一旦出现未见过的组合（如外来语、人名、缩写里的 `qx`）：

- 模型计算该转移概率 $P = 0$；
- 负对数似然损失直接爆炸：

$$
\text{Loss} = -\log(P) = -\log(0) \to +\infty
$$

**补丁：拉普拉斯平滑（Laplace Smoothing）**——给每个格子加一个虚拟计数值（伪计数）：

```python
P = (N + 1).float()  # 每一个格子都 + 1
P /= P.sum(1, keepdim=True)
```

- `+1` 后：原来 0 次的格子变 1 次，100 次变 101 次——概率永不为 0，`-log(0)` 永不出现；
- 平滑参数极大（如 `N + 10000`）时：频数差异被抹平，每行概率趋向均匀分布（每字母 $1/27$）。
- 这种人为「抹稀泥」防止模型过于极端的做法 = **平滑化（Smoothing）**。

### 2. 神经网络世界的对应物：L2 正则化（权重衰减）

神经网络没有频数矩阵，只有一个权重矩阵 $W$，用 $W$ 的大小来表达「自信程度」：

$$
\text{logits} = x @ W, \qquad \text{probs} = \text{softmax}(\text{logits})
$$

- 当 $W$ 全为 0 时：所有 logits 为 `[0, 0, ..., 0]` → $e^0 = 1$ → 概率全为 $\frac{1}{27} \approx 0.037$，即**绝对均匀分布**（极致平滑，谁也不偏向）。
- 当 $W$ 出现极大值（如 `10.0`、`-20.0`）时：Softmax 极度放大差距 → 某字符概率 `0.9999`，其余趋近 `0` → 极端自信，真实答案一旦偏离 Loss 暴涨。

**解法：在 Loss 后面追加惩罚项**

```python
# 原始的负对数似然损失 (NLL)
loss = -probs[torch.arange(num), ys].log().mean()

# 加上正则化惩罚项（L2 正则化）
loss = loss + 0.01 * (W**2).mean()
```

$$
\text{Total Loss} = \text{Data Loss} + \lambda \cdot \text{mean}(W^2)
$$

- 前半部分（Data Loss）：要求模型把训练集预测准；
- 后半部分（正则项 $\lambda \cdot \text{mean}(W^2)$）：**弹簧惩罚项**—— $W$ 变大则 $W^2$ 急剧增大，迫使优化器在「预测准」与「把 $W$ 往 0 压」之间权衡。
- 参数 $\lambda$（代码里的 `0.01`）：
  - $\lambda = 0$：无约束，权重可能学得巨大 → 死记硬背训练集（过拟合）；
  - $\lambda \to \infty$：弹簧拉力极大， $W$ 被压成全 0 → 退化为均匀分布。

### 3. 顿悟：两者的神圣对应

| 领域     | 做法                                | 极限情况（参数拉满）                          | 目的                   |
| ------ | --------------------------------- | ----------------------------------- | -------------------- |
| 纯统计学模型 | `(N + count) / sum`（平滑）           | `count` 极大时概率全变 $1/27$              | 避免概率为 0，防止 Loss 变无穷大 |
| 神经网络模型 | `loss + λ * (W**2).mean()`（L2 正则） | $\lambda$ 极大时 $W \to 0$，概率全变 $1/27$ | 约束权重大小，防止对训练集盲目自信    |

**一句话总结**：统计学里往计数上加 1（平滑），在神经网络里等价于给 Loss 加上权重平方惩罚（L2 正则化）。两者都在做同一件事——不让模型把话说得太绝，防止死记硬背，保留泛化余地。

### 4. 工业界映射（大厂面试高频）

#### 权重衰减（Weight Decay）与 AdamW

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
```

- `AdamW` 里的 `W` 即 **Weight Decay（权重衰减）**，本质就是「把权重往 0 压」的 L2 正则化。
- **面试必问：大模型训练为什么必须用 AdamW 而非传统 Adam？**
  - 传统 Adam 把 L2 正则放进梯度里一起更新，会被动量（Momentum）和自适应步长扭曲，无法对大权重有效衰减；
  - AdamW 将**权重衰减与梯度更新解耦**，实现精确的参数约束。

#### 标签平滑（Label Smoothing）

```python
loss = nn.CrossEntropyLoss(label_smoothing=0.1)
```

- 真实独热标签 `[0, 1, 0]` 若直接拟合非 0 即 1 的极端值，模型会为逼近 1.0 把 Logits 撑到几十上百 → 梯度消失 / 极端过拟合；
- `label_smoothing=0.1` 把标签平滑成 `[0.05, 0.9, 0.05]`，阻止极端 Logits，让注意力分布更平滑，提升鲁棒性。

**记忆锚点**：`loss = loss + 0.01 * (W**2).mean()` 就是一根给权重松绑 / 拉紧的弹簧。

<a id="interview-regularization"></a>

### 面试延伸：正则化与泛化

**问题 1：AdamW 与 Adam 加 L2 正则有什么区别？**

**回答：** 在 Adam 中把 $\lambda W$ 加进梯度后，它会被自适应二阶矩缩放，实际收缩强度因参数而异。AdamW 将权重衰减与梯度更新解耦，直接对参数做收缩，因此 weight decay 的语义和强度更可控；这也是大模型训练的常用选择。

**问题 2：所有参数都应该做 weight decay 吗？**

**回答：** 不一定。常见实践会把 bias 与 LayerNorm / RMSNorm 的缩放、平移参数排除，因为它们的作用不同于大权重矩阵，衰减它们未必有益。具体分组应说明为经验策略，最终需根据架构与验证结果确认。

**问题 3：Label Smoothing 解决什么问题，与 L2 是同一种正则吗？**

**回答：** Label Smoothing 直接软化监督目标，阻止分类器把正确类概率推到过度极端；L2 / weight decay 则约束参数大小。两者都能改善泛化和校准，但作用位置不同：一个改标签分布，一个改参数更新目标。

---

<a id="mini-batch-sgd"></a>

## 七、数据批处理：Mini-batch 机制与随机梯度下降 (SGD)

滑动窗口将一个词拆成多个“上下文 → 下一个字符”样本。以 `block_size = 3` 为例，训练集很容易扩展到数十万条；此时应以小批次而不是全量数据驱动每次参数更新。

<a id="batch-gradient-descent"></a>

### 1. 为什么不在全量数据集上做 Batch Gradient Descent (BGD)？

设训练集有 $N$ 个样本，单样本损失为 $\ell_i(\theta)$。全量目标和全量梯度为：

$$
L(\theta)=\frac{1}{N}\sum_{i=1}^{N}\ell_i(\theta),
\qquad
\nabla_\theta L(\theta)=\frac{1}{N}\sum_{i=1}^{N}\nabla_\theta\ell_i(\theta).
$$

- **吞吐与显存压力**：一次前向、反向传播必须保留全部样本的中间激活；样本、隐藏层或词表变大时，这会迅速成为内存瓶颈。
- **更新频率低**：必须处理完整个训练集后才走一步，单位时间内的参数更新次数很少。
- **缺少有益扰动**：全量梯度十分平滑；在复杂非凸损失面中，适量随机性常有助于离开鞍点或狭窄的次优区域。

<a id="mini-batch-principle"></a>

### 2. Mini-batch 的底层原理：用「方差」换取「速度」与「泛化」

每一步随机抽取大小为 $B$ 的批次 $\mathcal{B}$，用批次均值近似全量梯度：

$$
g_{\mathcal{B}}(\theta)=\frac{1}{B}\sum_{i\in\mathcal{B}}\nabla_\theta\ell_i(\theta),
\qquad
\theta_{t+1}=\theta_t-\eta\,g_{\mathcal{B}}(\theta_t).
$$

若样本是均匀随机抽取的，则该估计满足 $\mathbb{E}[g_{\mathcal{B}}]=\nabla_\theta L$：它在期望上是全量梯度的无偏估计。批次越小，方差通常越大；代价是更新方向更“吵”，收益是单步更轻、更新更多，并可能带来一定隐式正则化效果。常见的 $B=32,64,128$ 也较适合 GPU 的矩阵并行吞吐，但最终应由显存和实际吞吐测试决定。

```python
# Xtr: (N, block_size), Ytr: (N,)
batch_size = 32
ix = torch.randint(0, Xtr.shape[0], (batch_size,), device=Xtr.device)
Xb, Yb = Xtr[ix], Ytr[ix]  # Xb: (32, block_size), Yb: (32,)
```

> `torch.randint` 是有放回抽样；这对 SGD 很常见。若需要一个 epoch 内不重复地遍历样本，应使用 `torch.randperm` 再分批切片。

<a id="interview-minibatch"></a>

### 面试延伸：批处理与梯度噪声

**问题 1：为什么不使用全量 batch，也不总是使用 batch size 1？**

**回答：** 全量 batch 的激活显存和单步时延很高，更新次数少；单样本梯度噪声极大，GPU 矩阵并行利用率也差。mini-batch 在统计方差、硬件吞吐与显存之间折中，是现代训练的基本单位。

**问题 2：增大 batch size 后训练会发生什么？**

**回答：** 梯度估计通常更接近全量梯度、噪声更小，但每步消耗更多显存，更新次数也相对减少；学习率、warmup 和训练步数往往需要联动调节。不能只说“大 batch 一定更好”，应结合训练 token 总量、吞吐和验证效果判断。

**问题 3：梯度累积与真正的大 batch 有何关系？**

**回答：** 梯度累积将多个 micro-batch 的 loss / 梯度累加后再做一次 `optimizer.step()`，在显存不足时近似较大有效 batch。它减少单次激活显存，但无法完全等价于一次性大 batch：BatchNorm 的统计量、dropout 随机性、吞吐和通信时序都可能不同。

---

<a id="learning-rate"></a>

## 八、优化器生命线：学习率搜索与调度

<a id="lr-failure-modes"></a>

### 1. 学习率的作用与失效模式

学习率 $\eta$ 决定每次沿负梯度移动的步幅。它不是“越大越快”：

$$
\theta_{t+1}=\theta_t-\eta\nabla_\theta L(\theta_t).
$$

- $η$ **过大**：不断跨过谷底，损失剧烈震荡，甚至数值溢出为 `inf` / `nan`。
- $η$ **过小**：方向可能正确，但步长过短，训练进展缓慢、算力浪费严重。

<a id="lr-range-test"></a>

### 2. 对数尺度的 LR Range Test

学习率应按数量级搜索，而不是在线性坐标中均匀搜索。下面在 $10^{-3}$ 至 $10^0$ 间产生等间距的对数采样点：

```python
lre = torch.linspace(-3, 0, 1000)
lrs = 10.0 ** lre  # 0.001 ... 1.0；相邻候选值的比值恒定
```

Range Test 会逐步增大学习率并记录损失。选择标准是**损失稳定下降且下降较陡的区段**，通常应比损失最低点（往往已接近不稳定边界）再小一个量级左右。测试时必须在每步反传前清空梯度：

```python
lri, lossi = [], []
for i, lr in enumerate(lrs):
    ix = torch.randint(0, Xtr.shape[0], (32,), device=Xtr.device)
    logits = model(Xtr[ix])
    loss = F.cross_entropy(logits, Ytr[ix])

    for p in model.parameters():
        p.grad = None
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(p.grad, alpha=-lr.item())

    lri.append(lre[i].item())
    lossi.append(loss.item())
```

实际使用中，Range Test 的模型参数不应直接作为正式训练的起点；完成探索后应恢复初始权重，或重新初始化模型。

<a id="lr-decay"></a>

### 3. 学习率衰减（LR Decay）

训练初期参数离较优区域较远，较大的学习率利于快速搜索；后期则需要较小步长精细收敛。可采用分段衰减、余弦退火或带 warmup 的调度器。例如分段衰减：

$$
\eta_t=
\begin{cases}
0.1, & t < 100\,000,\\
0.01, & t \ge 100\,000.
\end{cases}
$$

降低学习率后，损失通常会进入新的、更低的平台；这不是“突然学会了新知识”，而是较小步长让参数能在原先震荡的谷底附近继续收敛。

<a id="interview-learning-rate"></a>

### 面试延伸：学习率与训练稳定性

**问题 1：为什么 Transformer 训练常使用 warmup？**

**回答：** 训练初期参数、激活与 Adam 的矩估计都尚未稳定，直接使用目标大学习率容易造成 loss 尖峰或数值不稳定。warmup 从较小学习率逐步升高，帮助模型进入稳定更新区间；之后常接余弦衰减或线性衰减。

**问题 2：训练 loss 震荡或出现 `nan`，如何排查？**

**回答：** 先检查学习率是否过大、输入和 loss 是否出现 `inf/nan`、混合精度的 loss scaling 是否正常，再查看梯度范数与激活范围。常见缓解措施包括降低学习率、增加 warmup、梯度裁剪、使用稳定的融合损失，以及排查数据异常；不应只靠盲目重跑。

**问题 3：学习率衰减为什么仍然有效？**

**回答：** 前期较大学习率便于跨过较差区域并快速降低 loss，后期较小步长降低震荡幅度，让参数在局部较优区域细致收敛。它改变的是优化的探索—收敛节奏，不是让模型凭空获得了更多知识。

---

<a id="mlp-language-model"></a>

## 九、MLP 字符语言模型：从上下文到交叉熵

这是固定窗口语言模型：使用前 $K$ 个字符预测下一个字符。令词表大小为 $V$、嵌入维度为 $D$、隐藏层维度为 $H$，其中本例 $K=3, D=2, H=100, V=27$。

<a id="mlp-forward"></a>

### 1. 前向数据流与核心公式

$$
\begin{aligned}
E &= C[X] &&\in \mathbb{R}^{B\times K\times D},\\
\tilde E &= \operatorname{reshape}(E) &&\in \mathbb{R}^{B\times(KD)},\\
H &= \tanh(\tilde E W_1+b_1) &&\in \mathbb{R}^{B\times H},\\
Z &= HW_2+b_2 &&\in \mathbb{R}^{B\times V},\\
\mathcal L &= \operatorname{CrossEntropy}(Z,Y).&&
\end{aligned}
$$

其中 $Z$ 是未归一化分数（logits），而非概率；`CrossEntropy` 会在内部完成稳定的 `log_softmax` 与负对数似然计算。

<a id="mlp-shapes"></a>

### 2. 张量形状索引

| 阶段 | 张量 | 形状 | 含义 |
| --- | --- | --- | --- |
| 输入批次 | `Xb` | `(B, 3)` | 每个样本有 3 个上下文字符索引 |
| 嵌入表 | `C` | `(27, 2)` | 每个词表项对应一个 2 维向量 |
| 查表结果 | `emb = C[Xb]` | `(B, 3, 2)` | 3 个字符各自的嵌入 |
| 展平特征 | `emb.view(B, 6)` | `(B, 6)` | 拼接上下文嵌入 |
| 隐藏层 | `h` | `(B, 100)` | `tanh` 后的非线性特征 |
| logits | `logits` | `(B, 27)` | 27 个候选下一个字符的分数 |

<a id="cross-entropy"></a>

### 3. 为什么使用 `F.cross_entropy`

手写 Softmax 与 NLL 的数学形式为：

$$
p_{i,j}=\frac{e^{z_{i,j}}}{\sum_{k=1}^{V}e^{z_{i,k}}},
\qquad
\mathcal L=-\frac{1}{B}\sum_{i=1}^{B}\log p_{i,y_i}.
$$

生产代码应使用融合算子，而不是直接执行 `logits.exp()`：

```python
loss = F.cross_entropy(logits, Yb)
```

其核心稳定变换为：

$$
\log\sum_j e^{z_j}=c+\log\sum_j e^{z_j-c},
\qquad c=\max_j z_j.
$$

减去行最大值不会改变 Softmax 概率，却能避免 `exp(100)` 一类上溢；融合实现也会减少中间张量和显存读写。对平均交叉熵，单个 logit 的梯度为 $\frac{\partial\mathcal L}{\partial z_{i,j}}=(p_{i,j}-\mathbb{1}[j=y_i])/B$。

<a id="interview-mlp-lm"></a>

### 面试延伸：Embedding 与语言模型

**问题 1：固定窗口 MLP 语言模型的主要局限是什么？**

**回答：** 它只能看到固定长度的最近上下文，窗口外信息完全丢失；若增大窗口，输入维度和第一层参数量会快速增长，也不能灵活处理长距离依赖。Transformer 使用自注意力，让每个 token 动态访问更长的上下文，因此成为现代 LLM 的主干。

**问题 2：为什么 embedding 是可学习的？它的梯度如何更新？**

**回答：** embedding 表是模型参数，每个 token id 只是选择其中一行。反向传播会将所有使用该 token 的位置梯度累加回对应行；相近的 token 是否形成相近向量完全由任务损失驱动学习，而不是由字符编号决定。

**问题 3：看到 `(B, T, D)`、`(B, T, V)` 时，如何快速判断张量含义？**

**回答：** 通常依次读为 batch、序列长度、特征或词表维度。`(B, T, D)` 多是 hidden states / embeddings，`(B, T, V)` 多是每个位置对词表的 logits；计算 token 级交叉熵时可把前两维展平为 `(B*T, V)`，标签展平为 `(B*T,)`。

---

<a id="data-split"></a>

## 十、数据集划分与泛化诊断

<a id="train-validation-test"></a>

### 1. 训练、验证与测试集

常用起点是 80% / 10% / 10%（实际比例应随数据规模调整）：

- **训练集（train）**：参与反向传播，用于更新 $C,W,b$ 等参数。
- **验证集（validation / dev）**：不更新参数；用于比较架构、学习率、隐藏层维度、正则化等超参数。
- **测试集（test）**：在方案完全确定前保持封存，只用于最终一次泛化评估。

测试集不是“第三个验证集”。根据测试分数反复改超参数，会将测试集信息泄漏进训练决策，使最终分数偏乐观。

<a id="fit-diagnosis"></a>

### 2. 欠拟合、良好拟合与过拟合

| 现象 | Train loss | Validation loss | 常见判断 |
| --- | --- | --- | --- |
| 欠拟合 | 高 | 高且接近训练集 | 容量不足、特征不足或训练不够 |
| 良好拟合 | 持续下降 | 略高于训练集且同步改善 | 泛化状态健康 |
| 过拟合 | 持续下降 | 停滞或回升 | 开始记忆训练样本中的噪声 |

出现过拟合时，可优先考虑更多数据、数据增强、权重衰减、早停或降低模型容量；不能用测试集曲线来选择这些策略。

<a id="interview-data-split"></a>

### 面试延伸：数据治理与评估

**问题 1：train、validation、test 的职责分别是什么？**

**回答：** train 用于更新参数；validation 用于选择超参数、模型结构与训练策略；test 在方案冻结后只用于最终泛化估计。若根据 test 分数反复调参，test 已参与决策，就不再是独立评估。

**问题 2：训练 loss 持续下降、验证 loss 回升，应该怎样处理？**

**回答：** 这是典型过拟合信号。优先检查数据量与数据质量，再尝试数据增强、正则化、早停、降低模型容量或调整训练时长；应以验证集选择方案，不能为了测试集分数反复试验。

**问题 3：MLLM 数据划分还要特别检查什么？**

**回答：** 除普通样本重复外，还要做图像近重复、文本近重复、图文对泄漏及训练集与公开 benchmark 的污染检查；同时按任务类型、语言、图像领域和难度分层评测。多模态模型的高分若来自数据泄漏，不能代表真实视觉—语言泛化能力。

---

<a id="mllm-training-map"></a>

## 十一、从 MLP 到多模态大模型的训练映射

本章 MLP 与现代 MLLM 的规模相差巨大，但训练闭环一致：输入特征经表示层进入网络，输出 logits，经交叉熵反传，再由优化器更新参数。

```text
图像 ──> ViT / Vision Encoder ──> 视觉特征 ──> Projector ─┐
                                                            ├─> Transformer ─> logits
文本 ──> Token Embedding ──────────────────────────────────┘
                                                               │
targets ───────────────────────────────────> CrossEntropy <───┘
                                                               │
                                                AdamW + LR Scheduler
```

对于自回归训练，常见张量形状为 `logits: (B, T, V)`、`targets: (B, T)`；计算 token 级交叉熵时需先合并批次与序列维：

```python
loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
```

在分布式训练中，Mini-batch 被分配给多个设备并行处理（DDP）；各设备梯度聚合后再更新。实践上常组合 AdamW、学习率 warmup / 余弦衰减，以及混合精度与梯度裁剪。视觉—文本投影和序列拼接的结构细节见第五章。

<a id="interview-mllm-training"></a>

### 面试延伸：MLLM 训练链路

**问题 1：请描述 decoder-only MLLM 的一次前向与 loss 计算。**

**回答：** 图像经视觉 encoder 产生视觉 token，再通过 projector 映射到 LLM hidden size；文本转为 token embedding 后，与视觉 token 按模板拼接为序列，经过 Transformer 生成 `(B, T, V)` logits。将需要预测的位置与 target 对齐，对 padding、图像占位符和非监督 prompt 位置做 mask，只对有效目标 token 计算交叉熵。

**问题 2：DDP 为什么能加速训练？每张卡上的参数会不一样吗？**

**回答：** DDP 将不同数据 shard 分给不同 GPU，各卡维护相同参数并独立前反传；随后通过 all-reduce 聚合梯度，各卡以相同梯度更新，因此参数保持一致。它提高的是数据并行吞吐，通信开销会随参数规模、网络带宽和梯度同步频率变化。

**问题 3：混合精度和梯度裁剪分别解决什么问题？**

**回答：** 混合精度以低精度存储和计算部分张量，降低显存并提高硬件吞吐，但需要处理小梯度下溢和关键操作精度；梯度裁剪在梯度范数过大时限制更新幅度，缓解训练尖峰或爆炸。两者常与 loss scaling、稳定 softmax 和合理学习率共同使用。

---

<a id="stable-training"></a>

## 十二、让训练稳下来：初始化、归一化与动量

前面已经见过训练循环：前向计算得到 loss，反向传播得到梯度，再由优化器更新参数。实际训练里，**梯度本身没错，训练仍可能不稳定**：刚开始 loss 特别大、激活值挤在边界、训练越久数值越飘，或者小批量梯度左右摇摆。本章把这些现象串成一条线：先管好开局，再管好训练过程，最后让更新走得更稳。

<a id="output-layer-init"></a>

### 1. 输出层：先让模型“谦虚地猜”

#### 现象：一开局就出现很高的 loss

在字符级 makemore 中，词表有 27 个字符。模型刚初始化时还没学到任何规律；理想状态是对每个字符一视同仁：

$$
p = \frac{1}{27}, \qquad -\ln p = -\ln\frac{1}{27} \approx 3.30.
$$

所以，开局的交叉熵在 `3.3` 左右其实很正常。若 Step 0 的 loss 是二三十甚至更高，通常不是“模型特别笨”，而是它**在毫无依据时过度自信**：随机的输出层权重和偏置让某些 logits 差得很远，Softmax 把这种差距放大；一旦高分给错了答案，负对数损失就会暴涨。这会让曲线前几十步像一根倒立的曲棍球杆（hockey stick）。

#### 做法：让开局的 logits 靠近 0

把最后一层偏置设为 0，并把输出权重缩小，就能让各类别 logits 起点相近：

```python
# 示意：不要直接修改已被优化器接管的参数；应在构造模型时初始化
b2 = torch.zeros(vocab_size)
W2 = torch.randn(n_hidden, vocab_size) * 0.01
```

此时 Softmax 看到的是一排接近 0 的数，输出接近均匀分布。这里的目标不是“让 loss 越小越好”，而是让模型从一个符合常识的基线出发：**还没学会之前，就老实地说不知道。**

<a id="activation-saturation"></a>

### 2. 隐藏层：避开 `tanh` 的饱和区

隐藏层若使用 `tanh`，它的输出范围是 `[-1, 1]`。问题不在于这个范围有限，而在于输入过大时，输出会贴死在 `-1` 或 `+1` 附近：

$$
\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x).
$$

例如 `tanh(x) ≈ 1` 时，导数就接近 0。根据链式法则，经过这里的梯度也会被乘得很小；这种单元并不是真的“死亡”，但它暂时几乎收不到学习信号。若直方图显示大量激活值堆在 `-1`、`+1` 两端，而中间很少，就说明输入 `tanh` 的数值尺度偏大。

解决思路很朴素：把前一层的权重尺度调到合适范围，让 `tanh` 的输入大多落在 0 附近。此处 `tanh` 更像一段近似直线，导数较大，信息和梯度都更容易通过。换句话说，初始化不仅决定 loss 的起点，也决定第一轮反向传播有没有路可走。

<a id="xavier-kaiming-init"></a>

### 3. 不再凭感觉缩放：Xavier 与 Kaiming 初始化

手写 `* 0.01` 能救急，但它不是普适答案。对于一层线性变换：

$$
y = \sum_{i=1}^{d_{in}} x_i w_i,
$$

在输入和权重近似独立、均值接近 0 的简化假设下：

$$
\operatorname{Var}(y) \approx d_{in}\operatorname{Var}(x)\operatorname{Var}(w).
$$

希望输出的尺度不要比输入越来越大或越来越小，就需要让权重方差随输入维度缩小。常见选择如下：

| 激活函数 / 场景 | 常见初始化 | 直觉 |
| --- | --- | --- |
| `tanh`、线性层 | Xavier / Glorot，量级约为 `1 / sqrt(fan_in)` | 同时尽量照顾前向激活和反向梯度的尺度 |
| ReLU / LeakyReLU | Kaiming / He，标准差约为 `sqrt(2 / fan_in)` | ReLU 会丢掉一部分信号，需要额外补偿 |

`fan_in` 是一个神经元接收的输入数。Kaiming 初始化的 `sqrt(2)` 并非魔法数字：它补偿了 ReLU 将负半轴置零后带来的尺度损失。需要注意，初始化只能保证**刚起跑时**的数值尺度大致健康；权重更新以后，分布还是会变化。

PyTorch 的 `nn.Linear` 已有合理默认初始化。学习时更值得掌握的是：换激活函数、堆很深的层，或发现激活/梯度明显爆炸或消失时，要检查初始化是否匹配，而不是盲目把所有权重乘一个常数。

<a id="batchnorm"></a>

### 4. BatchNorm：训练过程中的“实时校准”

初始化只管 Step 0；训练中每次更新权重，某一层输入的分布都可能继续漂移。Batch Normalization（BN）可以理解为在层间放了一个“实时校准器”。对一个 mini-batch 中某个特征维度，它计算均值与方差：

$$
\mu_B = \frac{1}{m}\sum_i x_i, \qquad
\sigma_B^2 = \frac{1}{m}\sum_i (x_i-\mu_B)^2,
$$

然后标准化并交还表达能力：

$$
\hat{x}_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}},
\qquad y_i=\gamma\hat{x}_i+\beta.
$$

其中 `ε` 防止除 0；`γ` 和 `β` 是可学习参数。它们很重要：BN 不是强迫所有层永远均值为 0、方差为 1，而是先把尺度稳定下来，再让模型自己决定应当放大、缩小或平移多少。

训练时 BN 用当前 batch 的统计量；推理时通常没有一批稳定的样本可统计，因此使用训练期间积累的 `running_mean`、`running_var`。这份“统计账本”是滑动平均得到的，稍后会和优化器动量区分开来。

<a id="ln-rmsnorm"></a>

### 5. Transformer 为什么更偏爱 LayerNorm / RMSNorm

BN 在 CNN 中很成功，但它会让一个样本的输出依赖同批其他样本：同一张图放进不同 batch，BN 的结果可能不同。对 Transformer / LLM，这很不方便：文本长度不同，需要 padding；推理经常是 batch size 1；大规模分布式训练还会让批统计与跨设备通信变得麻烦。

LayerNorm（LN）换了一个方向：它不在 batch 维度上把不同样本混在一起统计，而是对**每个 token 自己的 hidden dimension**计算均值和方差。因此它不依赖 batch 大小，也不需要训练、推理各用一套统计量。

RMSNorm 更进一步，通常只用均方根做缩放：

$$
\operatorname{RMSNorm}(x)=\frac{x}{\sqrt{\frac{1}{d}\sum_{j=1}^{d}x_j^2+\epsilon}}\odot g.
$$

它省掉了“减均值”这一步，计算更简单，实践中在许多现代 Transformer 中表现稳定。一个好记法：**BN 看同批同学，LN / RMSNorm 看 token 自己的一组特征。**

<a id="optimizer-momentum"></a>

### 6. 动量到底是什么：给更新方向加上惯性

SGD 的基本更新是：

$$
\theta_{t+1}=\theta_t-\eta g_t.
$$

这里的 $g_t$ 是当前 mini-batch 的梯度。因为每个 batch 只是全量数据的带噪声估计，方向会左右摆动。动量（optimizer momentum）不只听“这一拍”的梯度，而是保存一个速度 $v$，把最近多个方向做指数加权平均：

$$
v_t=\beta v_{t-1}+(1-\beta)g_t,
\qquad
\theta_{t+1}=\theta_t-\eta v_t.
$$

`β` 常取 `0.9`。直觉上像推一辆购物车：连续往同一方向推，车会越来越有惯性；偶然的一次横向干扰不会立刻把车带偏。于是动量通常能：

- 抑制 mini-batch 噪声造成的来回抖动；
- 在长期一致的下降方向上加速；
- 在狭长“山谷”中更快前进，而不是在两侧反复横跳。

但动量不是免费加速器。它会积累速度，因此学习率过大、突然改变数据分布或训练目标时，可能冲过头。常见组合是 `SGD(lr=..., momentum=0.9)`，并配合学习率衰减。Adam 也有类似思想：它的一阶矩 `exp_avg` 可以视为梯度动量，二阶矩则记录梯度大小以自适应调整步长。

<a id="two-momentums"></a>

### 7. 两种 Momentum，别混淆

“动量”在深度学习里常指两件不同的事，名字相同，作用位置不同：

| 名称 | 它在平均什么？ | 用来做什么？ |
| --- | --- | --- |
| 优化器动量（SGD / Adam） | 多个训练步的**梯度** | 让参数更新更平滑、更有方向感 |
| BatchNorm 的 momentum | 多个 batch 的**均值与方差** | 维护推理阶段可用的全局统计量 |

以 PyTorch BatchNorm 为例，`momentum=0.1` 的更新近似为：

$$
\text{running} \leftarrow 0.9\cdot\text{running}+0.1\cdot\text{batch\_stat}.
$$

这里的 `0.1` 是“本批统计量写进账本的比例”；而 SGD 的 `momentum=0.9` 通常表示“保留多少过去的更新方向”。两者的参数含义甚至方向相反，调参时不能望文生义地照搬。

**本章复习路线：** 输出层缩小，让开局不过度自信；匹配的初始化让激活与梯度能流动；归一化在训练中稳定特征尺度；优化器动量则稳定参数更新。它们都在处理“尺度和波动”，但分别作用在 logits、激活、特征统计和参数更新四个不同位置。

<a id="interview-stable-training"></a>

### 面试延伸：初始化、归一化与动量

**问题 1：Xavier 与 Kaiming 初始化如何选择？**

**回答：** 目标都是让多层网络开局时的激活和梯度尺度不过快放大或衰减。`tanh` 等近似对称激活常配 Xavier；ReLU / LeakyReLU 会截断负半轴，常配 Kaiming 的 $\sqrt{2/fan\_in}$ 尺度补偿。它们是合理起点，不保证训练全过程稳定。

**问题 2：为什么 LLM 通常使用 LayerNorm 或 RMSNorm，而不是 BatchNorm？**

**回答：** BN 依赖同 batch 样本的统计量，会产生跨样本耦合，训练与单样本推理还使用不同统计来源；变长序列 padding 和分布式同步也会带来麻烦。LN / RMSNorm 对每个 token 的 hidden 维度独立归一化，不依赖 batch size；RMSNorm 省去减均值，计算更简单。

**问题 3：SGD 的 momentum 与 BatchNorm 的 momentum 是同一个概念吗？**

**回答：** 不是。优化器动量平滑的是多步梯度，用于稳定并加速参数更新；BN 的 momentum 控制 running mean / variance 吸收当前 batch 统计量的比例，用于推理阶段统计。两者名字相同、默认数值含义却不同，调参时不能混用。

---

<a id="backprop-ninja"></a>

## 十三、Karpathy 反向传播课：从手推计算图到融合交叉熵

这节课围绕同一张计算图展开：字符索引经过 embedding 和两层 MLP 产生 logits，logits 经 softmax 与交叉熵变成一个 loss；随后 loss 的影响沿相反方向传回每个参数。核心不是记录许多 `d...` 变量，而是建立这条因果链：**参数改变 → logits 改变 → 概率改变 → loss 改变。反向传播只是在定量计算每一环“改变多少”。**

<a id="backprop-why-unpack"></a>

### 1. 为什么要把一行交叉熵拆开？

训练时常写成：

```python
loss = F.cross_entropy(logits, Yb)
```

这个算子同时完成了数值稳定处理、softmax、取正确类别的概率、取对数和 batch 平均。课程把它临时拆成“减最大值 → `exp` → 求和 → 除法 → `log` → 取标签 → 平均”，目的不是制造更长的训练代码，而是让计算图上每个节点都看得见。

一旦每个节点可见，就能逐个问出两个问题：它的输出如何依赖输入？它对 loss 的影响如何传回输入？把这些局部答案按反方向接起来，就是链式法则。推导完成后会发现，中间的许多项恰好抵消，最终应重新使用融合的 `F.cross_entropy`：它更稳定、更省显存、也更快。

<a id="backprop-forward-pipeline"></a>

### 2. 一次前向传播中，数据依次经历了什么？

反向传播绝不是凭空倒推；它只是把前向的每一步倒着走一遍。先把数据流看清楚，再讨论梯度。以字符级 MLP 为例，设：

- `B`：batch size，例如 32；
- `K`：上下文长度，例如 3 个字符；
- `D`：每个字符 embedding 的维度，例如 2；
- `H`：隐藏层宽度，例如 100；
- `V`：词表大小，例如 27。

前向过程与形状如下：

| 步骤 | 写法 | 形状 | 白话含义 |
| --- | --- | --- | --- |
| 查 embedding 表 | `emb = C[Xb]` | `(B, K, D)` | 给每个输入字符取一小段数字向量 |
| 拼成一行 | `embcat = emb.view(B, K*D)` | `(B, K·D)` | 把 3 个字符的向量首尾相连 |
| 第一层线性层 | `hpreact = embcat @ W1 + b1` | `(B, H)` | 每个隐藏神经元给上下文打分 |
| 激活 | `h = tanh(hpreact)` | `(B, H)` | 加入非线性能力 |
| 输出层 | `logits = h @ W2 + b2` | `(B, V)` | 给 27 个候选字符各给一个未归一化分数 |
| 交叉熵 | `loss = F.cross_entropy(logits, Yb)` | 标量 | 衡量预测分布与正确答案相差多远 |

这里有两个关键分界点：embedding 将离散的字符编号转为可学习的连续向量；logits 将隐藏特征转为每个候选字符的分数。`logits` 不是概率，可以是任意实数；softmax 才会把每一行 logits 变成和为 1 的概率。反向传播中的 `dlogits`，就是“每一个类别分数应该往上推还是往下拉、拉多少”。

<a id="backprop-what-is-gradient"></a>

### 3. 反向传播传递的到底是什么？

对任意中间变量 $x$，记：

$$
dx = \frac{\partial L}{\partial x}.
$$

它不是“$x$ 的导数”这种孤立概念，而是 **loss 对 $x$ 的敏感度**：其他量固定时，若把 $x$ 增加一个很小的量，loss 大约改变 $dx$ 倍。`dx > 0` 表示增大 $x$ 会让 loss 变大，梯度下降倾向于把 $x$ 往小调；`dx < 0` 则相反。

整张图的终点就是 loss 自己，因此从：

$$
dloss = \frac{\partial L}{\partial L}=1
$$

开始。每经过一个算子，已知的是它输出的梯度（上游梯度），目标是计算它每个输入的梯度。这个过程不是猜公式，而是反复使用同一条链式法则：

$$
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial out}
\frac{\partial out}{\partial x}.
$$

<a id="backprop-rules"></a>

### 4. 遇到不同算子时，梯度怎样分配？

从 loss 返回到输入的途中，复杂计算图最终由加、乘、求和、reshape、矩阵乘法等基础算子组成。下面四条规则覆盖了这节课最容易出错的部分：

1. **形状守恒。** 对任何张量 `x`，`dx.shape == x.shape`。这是最有效的手推调试器。
2. **链式法则。** 后面传来的梯度是“这一步输出对 loss 的影响”，再乘上“这一步输入对输出的影响”。
3. **分叉必累加。** 同一个变量被两条下游路径使用时，两条路径对 loss 的影响都算它的贡献，所以写 `+=`。这也是 PyTorch 每轮训练前要清空梯度的原因。
4. **广播必收拢。** 前向时一个数被复制给很多位置；反向时每个复制品的贡献都要加回原数，因此沿被广播的维度 `.sum(...)`。

一个很小的例子：`y = x + b`，其中 `x.shape=(B, H)`，而 `b.shape=(H,)`。前向时同一个 `b` 被加到了 B 行；若上游梯度为 `dy`，则：

$$
dx=dy, \qquad db=\sum_{\text{batch 维}}dy.
$$

这就是全连接层中偏置梯度要写 `db = dhpreact.sum(0)` 的原因，不是神秘技巧。

<a id="backprop-cross-entropy"></a>

### 5. 交叉熵如何一路反推回 logits？

先看课程把交叉熵拆开的稳定 softmax。对每个样本的一行 logits $z$：

$$
\begin{aligned}
m &= \max_j z_j,\\
n_j &= z_j-m,\\
c_j &= e^{n_j},\\
s &= \sum_j c_j,\\
p_j &= \frac{c_j}{s},\\
L &= -\log p_y.
\end{aligned}
$$

减去最大值 $m$ 不改变 softmax 概率，因为分子和分母都乘上同一个 $e^{-m}$；它只是避免 `exp(很大的数)` 溢出。这是数值稳定性，不是在改变模型含义。

现在从最后一步倒着看：

- `loss = -log(prob_of_correct_class).mean()`：正确类别位置得到负梯度，意思是“把正确类别概率抬高”；其他位置此时还是 0。`mean()` 会让每个样本的梯度除以 `B`。
- `p = c * (1/s)`：同一个概率同时依赖 `c` 和 `1/s`，因此梯度在这里分叉；乘法节点各自拿到“另一边 × 上游梯度”。
- `s = c.sum(...)`：一个总和由整行 `c` 相加得到，所以 `ds` 会复制回这一行每个 `c_j`；这与“前向求和、反向广播”成对。
- `c = exp(n)`：局部导数还是 `exp(n)`，也就是 `c` 自己，所以 `dn = dc * c`。
- `n = z - max(z)`：除了直接传回 `z` 的路径，`max` 也有一条路径。`max` 的梯度会分给最大元素（并列最大时是次梯度问题）。不过在 softmax + 交叉熵这套组合中，相关项最终会抵消；这也是融合推导比逐项硬记更有价值的地方。

把这些繁琐项化简，结论异常漂亮。对单个样本、类别 $j$：

$$
\frac{\partial L}{\partial z_j}=p_j-\mathbb{1}[j=y].
$$

其中 $\mathbb{1}[j=y]$ 是 one-hot 标签：正确类为 1，其余为 0。若 loss 是 batch 平均，再除以 $B$：

```python
# 只是帮助理解最终梯度，不是要求你在训练中手写它
dlogits = probs.clone()
dlogits[torch.arange(B), Yb] -= 1.0
dlogits /= B
```

这条公式可以这样读：

- 对错误类别：梯度是正的 `p_j`，梯度下降会降低其 logit；模型越错误地自信，压得越厉害。
- 对正确类别：梯度是 `p_y - 1`，通常为负，梯度下降会提高其 logit；已经接近 1 时，推动会自然变小。

所以它不是一条需要死背的孤立公式，而是一句很合理的话：**把预测概率往真实 one-hot 标签拉近。**

<a id="backprop-mlp-chain"></a>

### 6. logits 如何继续反推回 embedding 表？

有了 `dlogits`，后半段就是线性层与 `tanh` 的固定套路。先记住矩阵乘法的“形状配平法”。若：

$$
Y=XW,
$$

且 `dY` 已知，那么：

$$
dW=X^T dY, \qquad dX=dY W^T.
$$

它们不是凭空出现的：`dW` 必须和 `W` 同形状，`dX` 必须和 `X` 同形状；把矩阵维度对齐，转置的位置就自然确定了。

对输出层 `logits = h @ W2 + b2`：

$$
dW2=h^T dlogits, \qquad db2=\sum_{batch} dlogits, \qquad dh=dlogits\,W2^T.
$$

接下来过 `tanh`。既然 $h=\tanh(hpreact)$，局部导数为 $1-h^2$，因此：

$$
dhpreact=dh\odot(1-h^2).
$$

注意 $h$ 是前向时已经存下来的输出，所以不必重新计算 `tanh`。这也是反向传播为什么需要保存一些前向中间结果：用显存换计算时间。

对第一层 `hpreact = embcat @ W1 + b1` 再套一次同样的规则：

$$
dW1=embcat^T dhpreact, \qquad db1=\sum_{batch}dhpreact, \qquad dembcat=dhpreact\,W1^T.
$$

`embcat = emb.view(B, K·D)` 没有改变数字、只改变排布，所以反向只需把梯度 reshape 回 `(B, K, D)`。

最后是 embedding 查表：`emb = C[Xb]`。它可以理解为从表 `C` 中按字符编号复印若干行。某次查到了 `C[5]`，该位置的梯度就加回 `dC[5]`；如果字符 5 在一个 batch 中出现了 10 次，10 份贡献都必须相加。这正是“分叉必累加”在查表操作上的具体样子。实际框架会高效完成这一步，你不需要手写循环。

<a id="backprop-checking"></a>

### 7. 手推梯度的结果如何验证？

课程里常见的做法是：先让 PyTorch 对同一张计算图调用 `loss.backward()`，保存 `x.grad`；然后把自己的 `dx` 与它比较。这个习惯非常值得学，因为反向传播最容易犯的是“公式看似对、维度却错了”或“忘记累计某条路径”。

检查时可依次问：

1. `dx.shape` 是否等于 `x.shape`？
2. `dx` 与 `x.grad` 的最大绝对误差是否接近浮点误差（常见约 `1e-6` 到 `1e-8`，取决于 dtype）？
3. 是否把 batch 平均的 `/ B` 漏掉了？
4. 是否把广播的维度求和回来？是否漏了 `+=`？

这才是手推课的实际收益：当未来自己实现自定义算子、排查 `nan`、读论文中的损失函数或面试写简单梯度时，你有能力拆图验证；普通模型训练则放心交给 Autograd。

<a id="backprop-fusion"></a>

### 8. 为什么推导完成后反而要使用融合算子？

把交叉熵拆成许多小操作会产生 `norm_logits`、`counts`、`counts_sum`、`probs`、`logprobs` 等中间张量。每个张量都可能很大；训练时不仅前向要计算它们，反向通常还需要保留它们。对大 batch、长序列、大词表而言，这意味着更多显存读写、更慢的 GPU kernel 启动，以及 OOM 风险。

而数学已经告诉我们，Softmax 加交叉熵对 logits 的梯度直接就是 $p-y$。因此生产中应写：

```python
loss = F.cross_entropy(logits, Yb)
```

框架会在底层选择数值稳定且更少中间结果的实现。FlashAttention 的思想也相通：不是简单地“少写几行 Python”，而是让 Attention 的中间结果尽量留在 GPU 更快的片上存储中，避免反复写回显存；Triton 等工具则可用于编写这类融合 GPU 内核。共同原则是：**数学上能合并的连续操作，工程上就尽量少搬运中间数据。**

<a id="backprop-interview"></a>

### 9. MLLM 算法岗面试：会问什么，怎样回答？

这一课在算法岗面试中通常不会要求逐行手推几十个碎算子的梯度。考察重点是：能否抓住关键解析梯度、用形状解释矩阵反传，以及能否把数值稳定性和 GPU 性能联系到 LLM / MLLM 训练。下面的回答以“先给结论，再给原因，最后落到工程场景”为结构。

#### 问题 1：Softmax 与交叉熵对 logits 的梯度是什么？为什么？

**回答要点：** 对单样本，$\frac{\partial L}{\partial z_i}=p_i-y_i$；如果 loss 对 batch 取平均，还要除以 batch size $B$。其中 $p=\operatorname{softmax}(z)$，$y$ 是 one-hot 标签。

白板推导可写成两步。首先，softmax 的雅可比矩阵为：

$$
\frac{\partial p_k}{\partial z_i}=p_k(\mathbb{1}[k=i]-p_i).
$$

再代入 $L=-\sum_k y_k\log p_k$：

$$
\begin{aligned}
\frac{\partial L}{\partial z_i}
&=\sum_k\frac{\partial L}{\partial p_k}
\frac{\partial p_k}{\partial z_i}\\
&=\sum_k\left(-\frac{y_k}{p_k}\right)
p_k(\mathbb{1}[k=i]-p_i)\\
&=-y_i+p_i\sum_k y_k\\
&=p_i-y_i.
\end{aligned}
$$

**延伸到 MLLM：** 无论预测的是下一个文本 token、视觉离散 token，还是某个分类头的类别，最后一层若使用 softmax 交叉熵，核心梯度仍是“预测分布减真实分布”。这解释了为什么错得越自信，梯度修正越大。

#### 问题 2：线性层 $Y=XW+b$ 如何反向传播？

**回答要点：** 已知上游梯度 $dY$，有：

$$
dW=X^T dY,\qquad dX=dY W^T,\qquad db=\sum_{batch}dY.
$$

若 $X\in\mathbb{R}^{B\times d_{in}}$、$W\in\mathbb{R}^{d_{in}\times d_{out}}$、$dY\in\mathbb{R}^{B\times d_{out}}$，形状能直接验证：

$$
(d_{in}\times B)(B\times d_{out})=d_{in}\times d_{out},
$$

正好等于 $W$ 的形状；$dY W^T$ 的形状则正好等于 $X$。这套公式就是 Transformer 中 Q/K/V 投影、MLP 上下投影、视觉编码器投影层的共同底座。

#### 问题 3：为什么偏置梯度要在 batch 维度求和？

**回答要点：** 在 $Y=X+b$ 中，`b.shape=(d_out,)` 会在前向时广播成 `(B, d_out)`，同一个偏置被 B 个样本共同使用。每个样本都经由这一个 `b` 影响 loss，因此反向必须把 B 份贡献加回：`db = dY.sum(dim=0)`。

这个原则可概括为：**前向广播了哪个维度，反向就在该维度归约求和。**它也适用于 LayerNorm / RMSNorm 中的可学习缩放参数，只是归约维度要根据实际张量形状判断。

#### 问题 4：为什么 `F.cross_entropy` 比手写 `softmax().log()` 更适合训练？

**回答要点：** 有两层原因。

1. **数值稳定：** 框架在内部结合 `logsumexp` 计算，不直接产生可能上溢的 `exp(logit)`；因此输入应是 logits，而不是已经 softmax 的概率。
2. **算子融合：** 若拆成许多操作，每步都会产生并保存大中间张量，反向时还要读取它们。这会增加 HBM 显存读写、kernel 启动次数和激活显存，占用常常受内存带宽而非算力限制。

可以进一步把它联系到 FlashAttention：FlashAttention 不显式物化完整的注意力概率矩阵，而是按块在 GPU 的 SRAM 中完成在线 softmax 与矩阵乘法，必要时在反向重算部分结果。它降低 HBM 往返与激活内存，并非改变 Attention 的数学定义。对于长序列的 VLM / 多图 MLLM，这类内存优化尤其关键。

#### 问题 5：embedding 查表的梯度如何更新？

**回答要点：** `emb = C[token_ids]` 的前向是按 token id 从表 `C` 取行；反向则把每个位置的 `demb` 累加回被取到的那一行。相同 token 在 batch 或序列中出现多次时，梯度是这些位置贡献的和，这可视为 `scatter_add`。未出现的 token 行本轮梯度为 0。

**MLLM 延伸：** 文本 embedding、位置 embedding，以及视觉 token 经 projector 映射到语言模型 hidden space 后的梯度，都会沿各自的计算路径回传。需要区分：离散 token id 本身不可导；可学习的 embedding 表和 projector 权重可导。

#### 面试答题边界

- 能在白板推导 $p-y$、写出线性层梯度并解释形状，已经覆盖本课最常见的手推要求。
- 不必逐步推导 `exp`、倒数、`max` 等所有中间变量；若面试官追问，可说明它们通过链式法则展开，融合后化简为 $p-y$。
- 谈工程优化时，避免只说“融合更快”。应明确指出瓶颈是中间激活带来的 HBM 读写和显存占用，并说明 SRAM 分块计算、重计算与数值稳定 softmax 的作用。

**本课应沉淀的能力：** 看见一段网络代码时，能画出“输入 → 中间张量 → loss”的前向图；从 loss 倒着判断梯度形状、广播归约与分叉累加；再能说明为什么生产训练使用 Autograd 与融合内核。这套能力会直接迁移到 Transformer、ViT 和图文对齐 projector 的理解与调试中。

---

