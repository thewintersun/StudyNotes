## Self-Adaptive Network Pruning

论文地址：https://arxiv.org/abs/1910.08906

作者：Jinting Chen, Zhaocheng Zhu, Cheng Li, Yuming Zhao

机构：上海交大

发表：ICONIP2019



### 摘要

Deep convolutional neural networks have been proved successful on a wide range of tasks, yet they are still hindered by their large computation cost in many industrial scenarios. In this paper, we propose to reduce such cost for CNNs through a self-adaptive network pruning method (SANP). Our method introduces a general Saliency-and-Pruning Module (SPM) for each convolutional layer, which learns to predict saliency scores and applies pruning for each channel. Given a total computation budget, SANP adaptively determines the pruning strategy with respect to each layer and each sample, such that the average computation cost meets the budget. This design allows SANP to be more efficient in computation, as well as more robust to datasets and backbones. Extensive experiments on 2 datasets and 3 backbones show that SANP surpasses state-of-the-art methods in both classification accuracy and pruning rate.

### 介绍

这篇论文提出通过自适应网络剪枝方法（SANP）降低 CNN 的计算成本，通过对每个卷积层引入 Saliency-and-Pruning Module (SPM) 来实现，SPM 模块可以学习预测显著性分数，并对每个通道剪枝。SANP 会根据每个层和每个样本决定对应的剪枝策略。

根据下面的架构图，SPM 模块嵌入在卷积网络的每个层中。该模块可以基于输入特征去预测通道的显著性分数，然后为每个通道生成对应的剪枝决策。

对于剪枝决策为 0 的通道，则跳过卷积运算，然后利用分类目标和成本目标联合训练骨干网络和 SPM 模块。计算成本取决于每一层的剪枝决策。

![img](https://pic3.zhimg.com/80/v2-bbdab43a90c38157963230819c73183a_720w.jpg)

下表展示了该方法的一些结果：

![img](https://pic1.zhimg.com/80/v2-ed274f296efffb4f69057e0211ab2228_720w.jpg)