## Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts

论文地址：https://dl.acm.org/doi/abs/10.1145/3219819.3220007

作者：Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, Ed H. Chi 

机构：Google

发表：KDD 2018

https://zhuanlan.zhihu.com/p/55161704



### 摘要

基于神经的多任务学习已成功应用于许多现实世界的大规模应用，例如推荐系统。例如，在电影推荐中，除了为用户提供他们倾向于购买和观看的电影之外，系统还可能针对用户喜欢的电影进行优化。然而，多任务模型的预测质量往往对任务之间的关系很敏感。因此，研究特定任务目标和任务间之间关系的建模权衡非常重要。我们提出了一种新颖的多任务学习方法，多门专家混合 (Multi-gate Mixture-of-Experts， MMoE)。

In this work, we propose a novel multi-task learning approach, Multi-gate Mixture-of-Experts (MMoE), which explicitly learns to model task relationships from data. 

- We adapt the Mixture-of-Experts (MoE) structure to multi-task learning by sharing the expert submodels across all tasks, 
- while also having a gating network trained to optimize each task. 
- To validate our approach on data with different levels of task relatedness, we first apply it to a synthetic dataset where we control the task relatedness. We show that the proposed approach performs better than baseline methods when the tasks are less related. 
- We also show that the MMoE structure results in an additional trainability benefit, depending on different levels of randomness in the training data and model initialization. 

Furthermore, we demonstrate the performance improvements by MMoE on real tasks including a binary classification benchmark, and a large-scale content recommendation system at Google.



YouTube视频简析：[https://www.youtube.com/watch?v=Dweg47Tswxw](https://link.zhihu.com/?target=https%3A//www.youtube.com/watch%3Fv%3DDweg47Tswxw)

**核心观点：**

1. the Mixture-of- Experts (MoE) structure to multi-task learning by sharing the expert submodels across all tasks
2. explicitly models the task relationships and learns task-specific functionalities to leverage shared representations.
3. modulation and gating mechanisms can improve the trainability in training non-convex deep neural networks

**核心框架实现：**

![img](https://pic2.zhimg.com/80/v2-e782c1f1e1ae12bb5344f0a0d3353815_720w.jpg)

**Shared-bottom Multi-task Model：**共享底层网络：f(x), k 个任务各自有一个tower网络hk

![img](https://pic2.zhimg.com/80/v2-f69f47a8a8ba840bd7d090a8f38de461_720w.png)

Impact of Task Relatedness

1. 任务相关度减低，共享底层模型效果越差
2. 传统的多任务模型对任务相关度敏感

![img](https://pic4.zhimg.com/80/v2-560d87f9b834029312adebbc268e792f_720w.jpg)

The Original Mixture-of-Experts (MoE) Model

![img](https://pic3.zhimg.com/80/v2-f340dd396a34ab8db012303d5beaddd6_720w.jpg)

MoE Layer

1. The MoE layer has the same structure as the MoE model but accepts the output of the previous layer as input and outputs to a successive layer
2. For each input example, the model is able to select only a subset of experts by the gating network conditioned on the input

Multi-gate Mixture-of-Experts：每一个task独立的gate用来决定不同expert的结果的使用程度

![img](https://pic1.zhimg.com/80/v2-2045d3c5591bc0da4308b5a43ff18590_720w.jpg)

n为expert数量，w为n X d维矩阵

![img](https://pic4.zhimg.com/80/v2-aa30a8516a989631049197ba1ffad10f_720w.png)

Performance on Data with Different Task Correlations

1. 高相关度的任务多任务模型效果更好
2. MMoE模型在不同相关度多任务上的效果优于OMoE

![img](https://pic1.zhimg.com/80/v2-4ea5e096cf6a8ab6d0f8253ae2405778_720w.jpg)

Trainability：（这里的性能指的最终loss的分布）

1. 多次训练，shared-bottom 模型性能分布方差较大
2. 相关度为1是OMoE与MMoE性能近似，相关度为0.5时OMoE性能迅速下降
3. 三个模型最优性能接近一一致
   the histogram of the final loss values from repeated runs

![img](https://pic3.zhimg.com/80/v2-3dc5ed79ff645201c813543818d6513a_720w.jpg)

Census-income Data验证：

数据集:[https://archive.ics.uci.edu/ml/databases/census-income/](https://link.zhihu.com/?target=https%3A//archive.ics.uci.edu/ml/databases/census-income/)

实验设置(1)：Absolute Pearson correlation: 0.1768.

Task 1: Predict whether the income exceeds $50K;

Task 2: Predict whether this person’s marital status is never

实验设置(2)：Absolute Pearson correlation: 0.2373

Task1:Predictwhethertheeducationlevelisatleastcollege;

Task 2: Predict whether this person’s marital status is never married.

![img](https://pic4.zhimg.com/80/v2-a632959be8257edacb28c4765b006823_720w.jpg)

海量数据对比试验：只将shared-bottom顶层换成MMoE layer 不增加模型层数

两个任务：(1) predicting a user engagement related behavior;(2) predicting a user satisfaction related behavior

![img](https://pic1.zhimg.com/80/v2-15a7aaab11213d494177608659c045c4_720w.jpg)

Gate效果分析：

satisfaction subtask’s labels are sparser than the engagement sub- task’s, the gate for satisfaction subtask is more focused on a single expert.

![img](https://pic3.zhimg.com/80/v2-cf016ec3c0de22bf0cb49efff5b7f91e_720w.jpg)


开源实现地址：[https://github.com/drawbridge/keras-mmoe](https://link.zhihu.com/?target=https%3A//github.com/drawbridge/keras-mmoe)

思考应用方案：

1、按特征分expert

2、按结构分expert

3、task specific expert + common expert